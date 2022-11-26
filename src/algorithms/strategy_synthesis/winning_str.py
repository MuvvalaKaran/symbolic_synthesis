import re 
import sys
import time
import random

from functools import reduce
from collections import defaultdict
from typing import List, DefaultDict
from bidict import bidict

from cudd import Cudd, BDD

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import PartitionedDFA
from src.symbolic_graphs import DynamicFrankaTransitionSystem

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
                 ts_handle: DynamicFrankaTransitionSystem,
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

        # self.ts_transition_fun_list: List[BDD] = ts_handle.tr_state_bdds
        self.ts_transition_fun_list: List[List[BDD]] = ts_handle.sym_tr_actions
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
        self.ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        self.dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)

        self.sys_cube = reduce(lambda x, y: x & y, self.sys_act_vars)
        self.env_cube = reduce(lambda x, y: x & y, self.env_act_vars)

        # create ts and dfa combines cube
        self.prod_xlist = self.ts_x_list + self.dfa_x_list
        self.prod_xcube = reduce(lambda x, y: x & y, self.prod_xlist)

        # sys and env cube
        self.sys_env_cube = reduce(lambda x, y: x & y, [*self.sys_act_vars, *self.env_act_vars])

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

            # curr_act_cubes = list(curr_act.generate_cubes())
            curr_act_cubes = self.convert_prod_cube_to_func(dd_func=curr_act, prod_curr_list=self.sys_act_vars)


            # if multiple winning actions exisit from same state
            if len(curr_act_cubes) > 1:
                act_name = None
                while act_name is None:
                    # act_cube: List[int] = random.choice(curr_act_cubes)
                    # act_dd = self.manager.fromLiteralList(act_cube)
                    # using my cube generating function
                    act_dd: List[int] = random.choice(curr_act_cubes)
                    act_name: str = self.ts_bdd_sym_to_robot_act_map.get(act_dd, None)

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
        consist, strategy = transducer.solveEqn(self.sys_act_vars)
        # return strategy[0]
        # print("Particular solution:")
        # testing
        for g in strategy:
            p = g
            for u in self.sys_act_vars:
                p = p.compose(self.manager.bddOne(), u.index())
            # for u in y:
            #     p = p.compose(m.bddOne(), u.index())
            return p
            break 
            # p.display()
            # print(p)

        # state_action = strategy[0].vectorCompose(self.sys_act_vars, strategy[1])

        # restrict the strategy to non accpeting state, as we don not care what the Roboto after it reaches an accepting state
        if verbose:
            self.get_state_action(state_action & self.init_DFA)
        
        return state_action
    
    def get_pre_states(self, layer: int, alg_num: int) -> BDD:
        """
         Computes Predecessor States from the current set of states. 
        """
        # compute composed predecessor in one step
        if alg_num == 0:
            pre_prod_state: BDD = self.winning_states[layer].vectorCompose([*self.ts_x_list, *self.dfa_x_list],
                                                       [*self.ts_transition_fun_list, *self.dfa_transition_fun_list])
        # single vector compose
        elif alg_num == 1:
            pre_prod_state: BDD = self.manager.bddZero()
            for ts_transition in self.ts_transition_fun_list:
                pre_prod_state |= self.winning_states[layer].vectorCompose([*self.ts_x_list, *self.dfa_x_list],
                                                       [*ts_transition, *self.dfa_transition_fun_list])
        # two step vector compose
        elif alg_num == 2:
            # start = time.time()
            pre_prod_state: BDD = self.manager.bddZero()
            tmp = self.manager.bddZero()
            print("-------------------------------------------------------------------------------")
            for c, ts_transition in enumerate(self.ts_transition_fun_list):
                sa = time.time()
                tmp |= self.winning_states[layer].vectorCompose(self.ts_x_list, ts_transition)
                so = time.time()
                print(f"{self.ts_handle.tr_action_idx_map.inv[c]} predecssors: ", so - sa)
            print("-------------------------------------------------------------------------------")
            sa = time.time()
            pre_prod_state |= tmp.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
            so = time.time()

            # stop = time.time()
            print(f"Time to Compute DFA Predecessors: {so - sa}")
        elif alg_num == 3:
            pre_prod_state: BDD = self.manager.bddZero()
            for ts_transition in self.ts_transition_fun_list:
                tmp_list = [*ts_transition, *self.dfa_transition_fun_list]
                tmp_s = self.winning_states[layer]
                for count, var  in enumerate([*self.ts_x_list, *self.dfa_x_list]):
                    if not tmp_list[count].isZero():
                        tmp_s = tmp_s.compose(tmp_list[count], var.index())
                    else:
                        tmp_s = tmp_s.compose(~var, var.index())

                pre_prod_state |= tmp_s

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
            loop_start = time.time()
            if layer > 0 and stra_list[layer].compare(stra_list[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                if not ((self.init_TS & self.init_DFA) & stra_list[layer]).isZero():
                    print("A Winning Strategy Exists!!")
                    winning_str: BDD = self.get_strategy(transducer=stra_list[layer], verbose=True)

                    return stra_list[layer]
                else:
                    print("No Winning Strategy Exists!!!")
                    sys.exit(-1)

            # if verbose:
            print(f"**************************Layer: {layer}**************************")

            start = time.time()
            pre_prod_state = self.get_pre_states(layer=layer, alg_num=2)
            # pre_prod_state = self.get_pre_states(layer=layer, alg_num=2)

            # print(f"Pre state are same: {pre_prod_state_3 == pre_prod_state}")
            # print("**********************************")
            # pre_prod_state.display()
            # print("**********************************")
            # pre_prod_state_3.display()

            stop = time.time()
            print(f"Time to Compute Predecessors: {stop - start}")

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
                print("A Winning Strategy Exists!!")
                winning_str: BDD = self.get_strategy(transducer=stra_list[layer], verbose=True)

                return winning_str
                # return stra_list[layer + 1]

            # do existentail quantification
            self.winning_states[layer + 1] |=  stra_list[layer + 1].existAbstract(self.sys_cube)

            # print new winning states in each iteration
            if verbose:
                print(f"Winning states at Iter {layer + 1}")
                new_states = ~closed & pre_univ.existAbstract(self.sys_env_cube & self.ts_obs_cube)
                self.get_prod_states_from_dd(dd_func=new_states)

                closed |= pre_univ.existAbstract(self.sys_env_cube & self.ts_obs_cube)

            layer +=1
            loop_stop = time.time()
            print(f"Time to Complete One lop: {loop_stop - loop_start}")

