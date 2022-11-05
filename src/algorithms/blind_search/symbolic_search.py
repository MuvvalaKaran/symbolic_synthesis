'''
This file implements Symbolic Graph search algorithms
'''
import sys

from bidict import bidict
from functools import reduce
from cudd import Cudd, BDD, ADD

from typing import List
from itertools import product

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import SymbolicDFA, SymbolicTransitionSystem


class SymbolicSearch(BaseSymbolicSearch):
    """
    Given a Graph, find the shortest path as per the symbolic A* (BDDA*) algorithm as outlined by Jensen, Bryant, Valeso's paper.
    """

    def __init__(self,
                 ts_handle: SymbolicTransitionSystem,
                 dfa_handle: SymbolicDFA,
                 ts_curr_vars: list,
                 ts_next_vars: list,
                 ts_obs_vars: list,
                 dfa_curr_vars: list,
                 dfa_next_vars: list,
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state
        
        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_transition_fun_list = ts_handle.sym_tr_actions
        self.dfa_transition_fun = dfa_handle.dfa_bdd_tr
        self.ts_sym_to_curr_state_map: dict = ts_handle.predicate_sym_map_curr.inv
        self.ts_sym_to_S2obs_map: dict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_sym_map_curr.inv
        self.dfa_sym_to_nxt_state_map: dict = dfa_handle.dfa_predicate_sym_map_nxt.inv
        self.obs_bdd = ts_handle.sym_state_labels
        self.tr_action_idx_map = ts_handle.tr_action_idx_map

        # create corresponding cubes to avoid repetition
        self.ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        self.ts_ycube = reduce(lambda x, y: x & y, self.ts_y_list)
        self.ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)

        self.dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)

        # composed graph consists of state S, Z and hence are function TS and DFA bdd vars
        self.prod_xlist = self.ts_x_list + self.dfa_x_list
        self.prod_ylist = self.ts_y_list + self.dfa_y_list
        self.prod_xcube = reduce(lambda x, y: x & y, self.prod_xlist)
        self.prod_ycube = reduce(lambda x, y: x & y, self.prod_ylist)

        # composed monolithic TR
        self.composed_tr_list = self._construct_composed_tr_function()
    
    
    def _construct_composed_tr_function(self) -> List[BDD]:
        """
        A function that construct that conjoints the TR function of the TS and DFA along with S2P (state to obs BDD).

        Note: We prime the S2P BDD because we want to extract the next state in the DFA after we evolve over the TS.
        """

        obs_bdd_prime = self.obs_bdd.swapVariables(self.ts_x_list, self.ts_y_list) 
        composed_tr_list = []
        for tr_action in self.ts_transition_fun_list:
            composed_tr = tr_action & obs_bdd_prime & self.dfa_transition_fun
            composed_tr_list.append(composed_tr)
        
        return composed_tr_list
    

    def composed_symbolic_bfs_wLTL(self, verbose: bool = False, obs_flag: bool = False):
        """
        A function that compose the TR function from the Transition system and DFA and search symbolically over the product graph.

        Note: Since we are evolving concurrently over the prod DFA, the state labeling update is on step behind, i.e.,
         when we compute image over the 
        """
        # while open list is a sequence of bucket closed is a just a set of explored states and hence is not numbered
        open_list = []
        closed = self.manager.bddZero()

        # maintain a common layering number
        layer_num = 0
                
        # add the init state to ite respective DFA state. Note, we could start in some other state than the usual T0_init
        open_list.append(self.init_TS & self.init_DFA)

        while not self.target_DFA <= open_list[layer_num].existAbstract(self.ts_xcube):
            # remove all states that have been explored
            open_list[layer_num] = open_list[layer_num] & ~closed

            # If unexpanded states exist ... 
            if not open_list[layer_num].isZero():
                if verbose:
                    print(f"********************Layer: {layer_num}**************************")
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer_num]

                # compute the image of the TS states 
                prod_image_bdd = self.manager.bddZero()
                for prod_tr_action in self.composed_tr_list:
                    image_prod = self.image_per_action(trans_action=prod_tr_action,
                                                       From=open_list[layer_num],
                                                       xcube=self.prod_xcube,
                                                       x_list=self.prod_xlist,
                                                       y_list=self.prod_ylist)
                    
                    prod_image_bdd |= image_prod

                prod_image_bdd_restricted = prod_image_bdd.existAbstract(self.ts_obs_cube)
                
                if verbose:
                    self.get_states_from_dd(dd_func=prod_image_bdd, obs_flag=obs_flag)
                    
                open_list.append(prod_image_bdd_restricted)
            
            else:
                print("No plan exists! Terminating algorithm.")
                sys.exit(-1)
            
            layer_num += 1
        
        open_list[layer_num] = open_list[layer_num] & self.target_DFA

        if verbose:
            print("********************The goal state encountered is***********************")
            self.get_states_from_dd(dd_func=open_list[layer_num], obs_flag=obs_flag)

        print(f"Found a plan with least cost length {layer_num}, Now retireving it!")

        return self.retrieve_composed_symbolic_bfs(max_layer=layer_num, freach_list=open_list, verbose=verbose)
    

    def retrieve_composed_symbolic_bfs(self, max_layer: int, freach_list: dict, verbose: bool = False):
        """
        Retrieve the plan through Backward search by strarting from the Goal state and computing the interseaction of Forwards and Backwards
         Reachable set. 
        """
        g_layer = max_layer
        print("Working Retrieval plan now")

        current_prod = freach_list[g_layer]

        parent_plan = {}

        for g_layer in reversed(range(max_layer + 1)):
            new_current_prod = self.manager.bddZero()
            for tr_num, prod_tr_action in enumerate(self.composed_tr_list):
                pred_prod= self.pre_per_action(trans_action=prod_tr_action,
                                               From=current_prod,
                                               ycube=self.prod_ycube,
                                               x_list=self.prod_xlist,
                                               y_list=self.prod_ylist)

                if pred_prod.isZero():
                    continue
                
                if pred_prod & freach_list[g_layer - 1] != self.manager.bddZero():
                    # store the predecessor per action
                    tmp_current_prod = pred_prod & freach_list[g_layer - 1]
                    if verbose:
                        self.get_states_from_dd(dd_func=tmp_current_prod, obs_flag=False)
                    tmp_current_prod_res = (pred_prod & freach_list[g_layer - 1]).existAbstract(self.ts_obs_cube)
                    
                    self._append_dict_value_composed(parent_plan,
                                                     key_prod=tmp_current_prod_res,
                                                     action=self.tr_action_idx_map.inv[tr_num])
                    
                    new_current_prod |= tmp_current_prod_res 
            
            current_prod = new_current_prod
        
        return parent_plan