import re 
import sys
import random

from functools import reduce
from collections import defaultdict
from typing import List, DefaultDict, Union, Tuple
from bidict import bidict

from cudd import Cudd, BDD

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import PartitionedDFA
from src.symbolic_graphs import DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem

from utls import *


class ReachabilityGame(BaseSymbolicSearch):
    """
     A class that implements a Reachability algorithm to compute a set of winning region and the
     corresponding winning strategy for both the players in a Game G = (S, E). A winning strategy induces a play for the
     system player that satisfies the winning condition, i.e., to reach the accepting states on a symbolic given game.
    
    
     For a Reachability game the goal in terms of LTL could be written as : F(Accn States), i.e., Eventually
     reach the accepting region. A strategy for the sys ensures that it can force a visit to the accn states
     while the strategy for the env player is to remain in the trap region.
    """

    def __init__(self,
                 ts_handle: Union[DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem],
                 dfa_handle: PartitionedDFA,
                 ts_curr_vars: List[BDD],
                 dfa_curr_vars: List[BDD],
                 ts_obs_vars: List[BDD],
                 sys_act_vars: List[BDD],
                 env_act_vars: List[BDD],
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.ts_x_list = ts_curr_vars
        self.dfa_x_list = dfa_curr_vars

        self.sys_act_vars = sys_act_vars
        self.env_act_vars = env_act_vars

        self.ts_transition_fun_list: List[BDD] = ts_handle.tr_state_bdds
        self.dfa_transition_fun_list: List[BDD] = dfa_handle.tr_state_bdds

        self.ts_bdd_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        self.ts_bdd_sym_to_S2obs_map: bidict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_bdd_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_sym_map_curr.inv

        self.ts_bdd_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_bdd_sym_to_robot_act_map: bidict = ts_handle.predicate_sym_map_robot.inv

        self.obs_bdd: BDD = ts_handle.sym_state_labels

        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle

        self.winning_states: DefaultDict[int, BDD] = defaultdict(lambda: self.manager.bddZero())

        # create corresponding cubes to avoid repetition
        self.ts_xcube: BDD = reduce(lambda x, y: x & y, self.ts_x_list)
        self.dfa_xcube: BDD = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.ts_obs_cube: BDD = reduce(lambda x, y: x & y, self.ts_obs_list)

        self.sys_cube: BDD = reduce(lambda x, y: x & y, self.sys_act_vars)
        self.env_cube: BDD = reduce(lambda x, y: x & y, self.env_act_vars)

        # create ts and dfa combines cube
        self.prod_xlist: list = self.ts_x_list + self.dfa_x_list
        self.prod_xcube: BDD = reduce(lambda x, y: x & y, self.prod_xlist)

        # sys and env cube
        self.sys_env_cube = reduce(lambda x, y: x & y, [*self.sys_act_vars, *self.env_act_vars])

        # for bounded Game abstraction with addition boolean vars for # remaining human interventions
        if isinstance(self.ts_handle, BndDynamicFrankaTransitionSystem):
            self.max_hint: int = self.ts_handle.max_hint - 1
            self.hint_cube: BDD = reduce(lambda x, y: x & y, self.ts_handle.sym_vars_hint)
            self.hint_list = self.ts_handle.sym_vars_hint
            self.ts_bdd_sym_to_hint_map: bidict = self.ts_handle.predicate_sym_map_hint.inv

        self._initialize_w_t()
    
    def _initialize_w_t(self) -> None:
        """
         W : Set of winning states 
         T : Set of winning states and actions (winning strategies)
        
        This function initializes the set of winning states and set of winning states to the set of accepting states
        """
        ts_states: BDD = self.obs_bdd.existAbstract(self.ts_obs_cube)
        accp_states: BDD = ts_states & self.target_DFA

        self.winning_states[0] |= accp_states & self.obs_bdd
    
    @deprecated
    def get_state_action(self, dd_func: BDD, **kwargs) -> None:
        """
         A function  to print a strategy from each state in the dd_func

         This functionality is not working We need to fix it!
        """
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=dd_func,
                                                                     prod_curr_list=self.ts_x_list + self.dfa_x_list + self.sys_act_vars)
        for prod_cube in prod_cube_string:
            _ts_dd = prod_cube.existAbstract(self.dfa_xcube & self.ts_obs_cube & self.sys_cube)
            _ts_tuple = self.ts_bdd_sym_to_curr_state_map.get(_ts_dd)
            if _ts_tuple:
                _ts_name = self.ts_handle.get_state_from_tuple(state_tuple=_ts_tuple)
                assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

                _dfa_name = self._look_up_dfa_name(prod_dd=prod_cube.existAbstract(self.sys_cube),
                                                dfa_dict=self.dfa_bdd_sym_to_curr_state_map,
                                                ADD_flag=False,
                                                **kwargs)
                
                # get the sys action
                _sys_act_dd = prod_cube.existAbstract(self.ts_xcube & self.dfa_xcube & self.ts_obs_cube)
                _act_name = self.ts_bdd_sym_to_robot_act_map.get(_sys_act_dd)

                assert _act_name is not None, "Couldn't convert Strategy cube to its corresponding state-action pair. FIX THIS!!!"
                print(f"({_ts_name}, {_dfa_name})  -----> {_act_name} ")


    # overriding base class
    def get_prod_states_from_dd(self, dd_func: BDD, obs_flag: bool = False, **kwargs) -> None:
        """
         This base class overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=dd_func) 
        for prod_cube in prod_cube_string:
            _ts_dd = prod_cube.existAbstract(self.dfa_xcube & self.ts_obs_cube)
            _ts_tuple = self.ts_bdd_sym_to_curr_state_map.get(_ts_dd)
            _ts_name = self.ts_handle.get_state_from_tuple(state_tuple=_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=prod_cube,
                                               dfa_dict=self.dfa_bdd_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"({_ts_name}, {_dfa_name})")


    def roll_out_strategy(self, transducer: BDD, verbose: bool = False) -> None:
        """
         A function to roll out the synthesize winning strategy
        """

        curr_dfa_state = self.init_DFA
        curr_prod_state = self.init_TS & curr_dfa_state & self.obs_bdd
        counter = 0

        # until you reach a goal state. . .
        while (self.target_DFA & curr_prod_state).isZero():
            # get the next action
            curr_state_act: BDD =  transducer & curr_prod_state
            curr_act: BDD = curr_state_act.existAbstract(self.prod_xcube & self.ts_obs_cube)

            curr_act_cubes = list(curr_act.generate_cubes())
            # if multiple winning actions exisit from same state
            if len(curr_act_cubes) > 1:
                act_cube: List[int] = random.choice(curr_act_cubes)
                act_dd = self.manager.fromLiteralList(act_cube)
                act_name: str = self.ts_bdd_sym_to_robot_act_map[act_dd]

            else:
                act_name: str = self.ts_bdd_sym_to_robot_act_map[curr_act]

            if verbose:
                print(f"Step {counter}: {act_name}")
            
            # current state tuple
            curr_ts_state: BDD = curr_prod_state.existAbstract(self.dfa_xcube & self.ts_obs_cube & self.sys_env_cube)
            curr_ts_tuple: tuple = self.ts_bdd_sym_to_curr_state_map[curr_ts_state]

            # get add and del tuples
            for op in self.ts_handle.task.operators:
                if op.name == act_name:
                    add_tuple = self.ts_handle.get_tuple_from_state(op.add_effects)
                    del_tuple = self.ts_handle.get_tuple_from_state(op.del_effects)
                    break

            # construct the tuple for next state
            next_tuple = list(set(curr_ts_tuple) - set(del_tuple))
            next_tuple = tuple(sorted(list(set(next_tuple + list(add_tuple)))))

            # look up its corresponding formula
            curr_ts_state: BDD = self.ts_bdd_sym_to_curr_state_map.inv[next_tuple]

            curr_ts_lbl = curr_ts_state & self.obs_bdd
            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state in self.dfa_bdd_sym_to_curr_state_map.keys():
                dfa_pre = dfa_state.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
                edge_exists: bool = not (dfa_pre & (curr_dfa_state & curr_ts_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state = dfa_state
                    break 

            # curr_prod_state = curr_ts_state & self.init_DFA
            curr_prod_state = curr_ts_state & curr_dfa_state

            counter += 1


    def get_strategy(self, transducer: BDD, verbose: bool = False) -> BDD:
        """
         A function that compute the action to take from each state from the transducer. 
        """
        strategy: BDD = transducer.solveEqn(self.sys_act_vars)
        state_action = strategy[0].vectorCompose(self.sys_act_vars, strategy[1])


        # restrict the strategy to non accpeting state, as we don not care what the Roboto after it reaches an accepting state
        if verbose:
            self.get_state_action(state_action & self.init_DFA)
        
        return state_action

    def get_pre_states(self, layer: int) -> BDD:
        """
         A function to compute all predecessors from the current set of winning states
        """
        pre_prod_state: BDD = self.winning_states[layer].vectorCompose([*self.ts_x_list, *self.dfa_x_list],
                                                       [*self.ts_transition_fun_list, *self.dfa_transition_fun_list])
        
        return pre_prod_state
    
    def solve(self, verbose: bool = False) -> BDD:
        """
         This function compute the set of winning states and winnign strategies. 
        """

        stra_list = defaultdict(lambda: self.manager.bddZero())
        closed = self.manager.bddZero()  # BDD to keep track of winning states explore till now

        layer: int = 0

        stra_list[layer] = self.winning_states[layer]
        if verbose:
            closed |= self.winning_states[layer].existAbstract(self.ts_obs_cube)

        while True:
            if layer > 0 and stra_list[layer].compare(stra_list[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                if not ((self.init_TS & self.init_DFA) & stra_list[layer]).isZero():
                    print("A Winning Strategy Exists!!")
                    # winning_str: BDD = self.get_strategy(transducer=stra_list[layer], verbose=True)

                    return stra_list[layer]
                else:
                    print("No Winning Strategy Exists!!!")
                    sys.exit(-1)

            print(f"**************************Layer: {layer}**************************")
            
            pre_prod_state = self.get_pre_states(layer=layer)
            
            # we need to fix the state labeling
            pre_prod_state = pre_prod_state.existAbstract(self.ts_obs_cube)

            # do universal quantification
            pre_univ = (pre_prod_state).univAbstract(self.env_cube)
            # add the correct labels back
            pre_univ = pre_univ & self.obs_bdd

            # remove self loops
            stra_list[layer + 1] |= stra_list[layer] | (~self.winning_states[layer] & pre_univ)
            
            # if init state is reached
            if not ((self.init_TS & self.init_DFA) & stra_list[layer + 1]).isZero():
                print(f"************************** Reached the Initial State at layer: {layer + 1} **************************")
                print("A Winning Strategy Exists!!")
                # winning_str: BDD = self.get_strategy(transducer=stra_list[layer], verbose=True)

                # return winning_str
                if verbose:
                    print(f"Winning states at Iter {layer + 1}")
                    new_states = ~closed & pre_univ.existAbstract(self.sys_env_cube & self.ts_obs_cube)
                    self.get_prod_states_from_dd(dd_func=new_states)

                    closed |= pre_univ.existAbstract(self.sys_env_cube & self.ts_obs_cube)
                return stra_list[layer + 1]

            # do existentail quantification
            self.winning_states[layer + 1] |=  stra_list[layer + 1].existAbstract(self.sys_cube)

            # print new winning states in each iteration
            if verbose:
                print(f"Winning states at Iter {layer + 1}")
                new_states = ~closed & pre_univ.existAbstract(self.sys_env_cube & self.ts_obs_cube)
                self.get_prod_states_from_dd(dd_func=new_states)

                closed |= pre_univ.existAbstract(self.sys_env_cube & self.ts_obs_cube)

            layer +=1


class BndReachabilityGame(ReachabilityGame):
    """
     Compute the Winning strategy for Game Abstraction with bounded human intervention encounded as counter to each Node. 
    """

    def __init__(self,
                 ts_handle: Union[DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem],
                 dfa_handle: PartitionedDFA,
                 ts_curr_vars: List[BDD],
                 dfa_curr_vars: List[BDD],
                 ts_obs_vars: List[BDD],
                 sys_act_vars: List[BDD],
                 env_act_vars: List[BDD],
                 cudd_manager: Cudd):
        super().__init__(ts_handle,
                         dfa_handle,
                         ts_curr_vars,
                         dfa_curr_vars,
                         ts_obs_vars,
                         sys_act_vars,
                         env_act_vars,
                         cudd_manager)
    
    def get_pre_states(self, layer: int) -> BDD:
        """
         We  have an additional human intervention variables that we need to account 
        """
        pre_prod_state: BDD = self.winning_states[layer].vectorCompose([*self.ts_x_list, *self.ts_handle.sym_vars_hint, *self.dfa_x_list],
                                                       [*self.ts_transition_fun_list, *self.dfa_transition_fun_list])
        
        return pre_prod_state


    def get_prod_states_from_dd(self, dd_func: BDD, obs_flag: bool = False, **kwargs) -> None:
        """
         This base class overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """

        prod_curr_list=self.ts_x_list + self.dfa_x_list + self.hint_list
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=dd_func, prod_curr_list=prod_curr_list) 
        for prod_cube in prod_cube_string:
            _ts_dd = prod_cube.existAbstract(self.dfa_xcube & self.ts_obs_cube & self.hint_cube)
            _ts_tuple = self.ts_bdd_sym_to_curr_state_map.get(_ts_dd)
            _ts_name = self.ts_handle.get_state_from_tuple(state_tuple=_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=prod_cube.existAbstract(self.hint_cube),
                                               dfa_dict=self.dfa_bdd_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            _ts_hint_dd = prod_cube.existAbstract(self.dfa_xcube & self.ts_obs_cube & self.ts_xcube)
            _ts_hint: int = self.ts_bdd_sym_to_hint_map.get(_ts_hint_dd)
            
            print(f"([{_ts_name},{_ts_hint}], {_dfa_name})")


    def evolve_as_per_human(self, curr_state_tuple: tuple, curr_dfa_state: BDD, curr_hint: int, ract_name: str) -> Tuple[BDD, tuple]:
        """
         A function that compute the next state tuple given the current state tuple. 
        """
        nxt_state_tuple = random.choice(self.ts_handle.adj_map[curr_state_tuple][ract_name][curr_hint]['h'])
        # look up its corresponding formula
        nxt_ts_state: BDD = self.ts_bdd_sym_to_curr_state_map.inv[nxt_state_tuple]

        # update the DFA state as per the human move.
        nxt_ts_lbl = nxt_ts_state & self.obs_bdd
        
        # create DFA edge and check if it satisfies any of the dges or not
        for dfa_state in self.dfa_bdd_sym_to_curr_state_map.keys():
            dfa_pre = dfa_state.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
            edge_exists: bool = not (dfa_pre & (curr_dfa_state & nxt_ts_lbl)).isZero()

            if edge_exists:
                nxt_dfa_state = dfa_state
                break
            
        return nxt_ts_lbl & nxt_dfa_state, nxt_state_tuple
    

    def human_intervention(self, ract_name: str, curr_state_tuple: tuple, curr_dfa_state: BDD, curr_hint: int, verbose: bool = True) -> Tuple[tuple, int]:
        """
         Evolve on the game as per human intervention
        """
        # get the next action
        hnext_tuple = curr_state_tuple
        del_pred: bool = False
        # # if curr robot action is transit or transfer then remove to-loc and to-obj predicate respectively for human int. 
        if 'transit' in ract_name:
            for tidx, pred in enumerate(self.ts_handle.get_state_from_tuple(hnext_tuple)):
                if 'to-obj' in pred:
                    del_pred = True
                    break
            
        elif 'transfer' in ract_name:
            for tidx, pred in enumerate(self.ts_handle.get_state_from_tuple(hnext_tuple)):
                if 'to-loc' in pred:
                    del_pred = True
                    break
        
        if del_pred:
            # remove to-obj prediacte
            hnext_tuple = set(hnext_tuple) - set([list(curr_state_tuple)[tidx]]) 
            hnext_tuple = tuple(sorted(list(hnext_tuple))) 
        
        itr = 0
        if len(self.ts_handle.adj_map[hnext_tuple][ract_name][curr_hint]['h']) > 0:
            nxt_prod_state, nxt_ts_tuple = self.evolve_as_per_human(curr_state_tuple=hnext_tuple,
                                                                    curr_hint=curr_hint,
                                                                    curr_dfa_state=curr_dfa_state,
                                                                    ract_name=ract_name)
            # if not (self.target_DFA & curr_prod_state).isZero():
            #     break
            # forcing human to not make a move that satisfies the specification
            while not (self.target_DFA & nxt_prod_state).isZero():
                nxt_prod_state, nxt_ts_tuple = self.evolve_as_per_human(curr_state_tuple=hnext_tuple,
                                                                        curr_hint=curr_hint,
                                                                        curr_dfa_state=curr_dfa_state,
                                                                        ract_name=ract_name)
                # hacky way to avoid infinite looping
                itr += 1
                if itr > 5:
                    break

            if verbose:
                print(f"Human Intervened: New Conf. {self.ts_handle.get_state_from_tuple(nxt_ts_tuple)}")

            # reduce the human int count by 1
            curr_hint -= 1
            curr_hint_dd = self.ts_bdd_sym_to_hint_map.inv[curr_hint]

            # update hint counter 
            curr_prod_state = nxt_prod_state & curr_hint_dd
            curr_ts_tuple = nxt_ts_tuple

        return curr_ts_tuple, curr_hint


    def roll_out_strategy(self,
                          transducer: BDD,
                          verbose: bool = False) -> None:
        """
         A function to roll out the synthesize winning strategy
        """

        curr_dfa_state = self.init_DFA
        curr_prod_state = self.init_TS & curr_dfa_state & self.obs_bdd
        curr_hint_dd: BDD = self.init_TS.existAbstract(self.ts_xcube)
        curr_hint: int = self.ts_bdd_sym_to_hint_map[curr_hint_dd]
        counter = 0
        ract_name: str = ''

        # until you reach a goal state. . .
        while (self.target_DFA & curr_prod_state).isZero():
            # current state tuple
            curr_ts_state: BDD = curr_prod_state.existAbstract(self.dfa_xcube & self.ts_obs_cube & self.sys_env_cube & self.hint_cube)
            curr_ts_tuple: tuple = self.ts_bdd_sym_to_curr_state_map[curr_ts_state]

            curr_state_act: BDD =  transducer & curr_prod_state
            curr_act: BDD = curr_state_act.existAbstract(self.prod_xcube & self.ts_obs_cube & self.hint_cube)

            # curr_act_cubes = list(curr_act.generate_cubes())
            curr_act_cubes = self.convert_prod_cube_to_func(dd_func=curr_act, prod_curr_list=self.sys_act_vars)

            # if multiple winning actions exisit from same state
            if len(curr_act_cubes) > 1:
                ract_name = None
                while ract_name is None:
                    ract_dd: List[int] = random.choice(curr_act_cubes)
                    ract_name = self.ts_bdd_sym_to_robot_act_map.get(ract_dd, None)

            else:
                ract_name = self.ts_bdd_sym_to_robot_act_map[curr_act]

            if verbose:
                print(f"Step {counter}: {ract_name}; Hint: {curr_hint}")
            
            # get add and del tuples
            for op in self.ts_handle.task.operators:
                if op.name == ract_name:
                    add_tuple = self.ts_handle.get_tuple_from_state(op.add_effects)
                    del_tuple = self.ts_handle.get_tuple_from_state(op.del_effects)
                    break

            # construct the tuple for next state
            next_tuple = list(set(curr_ts_tuple) - set(del_tuple))
            next_tuple = tuple(sorted(list(set(next_tuple + list(add_tuple)))))

            # get human intervention.
            if curr_hint > 0:
                # coin = random.randint(0, 1)
                heads = 1
                if heads:
                    next_tuple, curr_hint = self.human_intervention(ract_name=ract_name, curr_state_tuple=curr_ts_tuple, curr_dfa_state=curr_dfa_state, curr_hint=curr_hint, verbose=True)

            # look up its corresponding formula
            curr_ts_state: BDD = self.ts_bdd_sym_to_curr_state_map.inv[next_tuple]

            curr_ts_lbl = curr_ts_state & self.obs_bdd
            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state in self.dfa_bdd_sym_to_curr_state_map.keys():
                dfa_pre = dfa_state.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
                edge_exists: bool = not (dfa_pre & (curr_dfa_state & curr_ts_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state = dfa_state
                    break 

           
            curr_prod_state = curr_ts_state & curr_dfa_state & self.ts_bdd_sym_to_hint_map.inv[curr_hint]

            counter += 1