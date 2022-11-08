import re 
import sys

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

        self.ts_transition_fun_list: List[BDD] = ts_handle.tr_state_bdds
        self.dfa_transition_fun_list: List[BDD] = dfa_handle.tr_state_bdds

        self.ts_bdd_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        self.ts_bdd_sym_to_S2obs_map: bidict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_bdd_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_sym_map_curr.inv

        self.ts_bdd_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_bdd_sym_to_robot_act_map: bidict = ts_handle. predicate_sym_map_robot.inv

        self.obs_bdd: BDD = ts_handle.sym_state_labels

        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle

        self.winning_states: DefaultDict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        self.winning_str:  DefaultDict[int, BDD] = defaultdict(lambda: self.manager.bddZero())

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
        self.winning_str[0] |= accp_states
    
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

    @deprecated
    def roll_out_strategy(self):
        """
         A function to roll out the synthesize winning strategy
        """

        curr_pod_state = self.init_TS & self.init_DFA
        # until you reach a goal state. . .
        while (self.target_DFA & curr_pod_state).isZero():
            # get the next action 

            # how to get successor?
            raise NotImplementedError()


    def get_strategy(self, transducer: BDD, verbose: bool = False) -> BDD:
        """
         A function that compute the action to take from each state from the transducer. 
        """
        strategy: BDD = transducer.solveEqn(self.sys_act_vars)
        state_action: BDD = strategy[1][0].existAbstract(self.ts_obs_cube)

        # restrict the strategy to non accpeting state, as we don not care what the Roboto after it reaches an accepting state
        if verbose:
            self.get_state_action(state_action.restrict(~self.dfa_x_list[0]))
        
        return state_action


    def solve(self, verbose: bool = False) -> BDD:
        """
         This function compute the set of winning states and winnign strategies. 
        """

        stra_list = defaultdict(lambda: self.manager.bddZero())
        # closed = self.manager.bddZero()

        layer: int = 0

        stra_list[layer] = self.winning_states[layer]

        while True:
            # remove all states that have been explored
            # open_list[layer] = open_list[layer] & ~closed
            if layer > 0 and stra_list[layer].compare(stra_list[layer - 1], 2):
                print("**************************Reached a Fixed Point**************************")
                if not ((self.init_TS & self.init_DFA) & stra_list[layer]).isZero():
                    print("A Winning Strategy Exists!!")
                    # winning_str: BDD = self.get_strategy(transducer=stra_list[layer], verbose=False)
                    
                    # # testing print action from init state
                    # print("init action:")
                    # act = winning_str & (self.init_TS & self.init_DFA)
                    # print(act)

                    return stra_list[layer]
                else:
                    print("No Winning Strategy Exists!!")
                    return self.manager.bddZero()

            # If unexpanded states exist ... 
            # if not open_list[layer].isZero():
            if verbose:
                print(f"**************************Layer: {layer}**************************")

            pre_prod_state: BDD = self.winning_states[layer].vectorCompose([*self.ts_x_list, *self.dfa_x_list],
                                                       [*self.ts_transition_fun_list, *self.dfa_transition_fun_list])

            # pre_ts_state: BDD = stra_list[layer].vectorCompose(self.ts_x_list, self.ts_transition_fun_list)
            # pre_ts_lbl: BDD = pre_ts_state & self.obs_bdd
            # pre_dfa_state: BDD = pre_ts_lbl.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)

            # if verbose:
            #     self.get_prod_states_from_dd(dd_func=(pre_prod_state).existAbstract(self.sys_env_cube), obs_flag=False)


            # do universal quantification
            # pre_univ = (pre_ts_lbl & pre_dfa_state).univAbstract(self.env_cube)
            pre_univ = (pre_prod_state).univAbstract(self.env_cube)

            # self.get_prod_states_from_dd(dd_func=pre_univ.existAbstract(self.sys_env_cube))
            
            # remove self loops
            stra_list[layer + 1] |= stra_list[layer] | (~self.winning_states[layer] & pre_univ)

            # if verbose: 
            #     print(f"Winning strategy at Iter {layer + 1}")
            #     self.get_state_action(dd_func=stra_list[layer + 1])

            # do existentail quantification
            self.winning_states[layer + 1] |=  stra_list[layer + 1].existAbstract(self.sys_cube & self.ts_obs_cube)

            if verbose:
                print(f"Winning states at Iter {layer + 1}")
                self.get_prod_states_from_dd(dd_func= (~self.winning_states[layer] & pre_univ).existAbstract(self.sys_cube & self.ts_obs_cube))

            layer +=1


