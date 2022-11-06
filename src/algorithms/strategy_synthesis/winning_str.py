import re 
import sys

from functools import reduce
from collections import defaultdict
from typing import List, DefaultDict
from bidict import bidict

from cudd import Cudd, BDD

from src.symbolic_graphs import PartitionedDFA
from src.symbolic_graphs import DynamicFrankaTransitionSystem


class ReachabilityGame():
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
        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state
        self.ts_obs_list = ts_obs_vars

        self.ts_x_list = ts_curr_vars
        self.dfa_x_list = dfa_curr_vars

        self.sys_act_vars = sys_act_vars
        self.env_act_vars = env_act_vars

        self.ts_transition_fun_list = ts_handle.tr_state_bdds
        self.dfa_transition_fun = dfa_handle.tr_state_bdds

        self.ts_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        self.ts_sym_to_S2obs_map: bidict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_sym_map_curr.inv

        self.ts_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_sym_to_robot_act_map: bidict = ts_handle. predicate_sym_map_robot.inv

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

        self.manager = cudd_manager

        self._initialize_w_t()
    
    def _initialize_w_t(self) -> None:
        """
         W : Set of winning states 
         T : Set of winning states and actions (winning strategies)
        
        This function initializes the set of winning states and set of winning states to the set of accepting states
        """
        ts_states: BDD = self.obs_bdd.existAbstract(self.ts_obs_cube)
        accp_states: BDD = ts_states & self.target_DFA

        self.winning_states[0] |= accp_states
        self.winning_str[0] |= accp_states


    def solve(self, verbose: bool = False):
        """
         This function compute the set of winning states and winnign strategies. 
        """

        open_list = defaultdict(lambda: self.manager.bddZero())
        closed = self.manager.bddZero()

        layer: int = 0

        open_list[0] = self.winning_states[0]

        while not open_list[layer].isZero():
            # remove all states that have been explored
            open_list[layer] = open_list[layer] & ~closed

            # If unexpanded states exist ... 
            if not open_list[layer].isZero():
                if verbose:
                    print(f"********************Layer: {layer + 1}**************************")
                
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer]

                pre_prod_state: BDD = open_list[layer].vectorCompose([*self.ts_x_list, *self.dfa_x_list],
                                                           [*self.ts_transition_fun_list, *self.dfa_transition_fun])
                
                
                # do universal quantification

                # do existentail quantification

                pass


