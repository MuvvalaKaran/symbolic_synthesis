import re
import sys

from functools import reduce
from typing import List, Union

from cudd import Cudd, BDD, ADD
from src.symbolic_graphs import SymbolicAddDFA, SymbolicWeightedTransitionSystem
from src.algorithms.base import BaseSymbolicSearch


class MultipleFormulaDijkstra(BaseSymbolicSearch):
    """
    Given a Transition systenm, and n DFAs associated with different Formulas, this class computes the minimum cost path
    by searching over the composed graph using the Symbolic Dijkstras algorithm.

    Algorithm inspired from Peter Kissmann's PhD thesis - Symbolic Search in Planning and General Game Playing.
     Link - https://media.suub.uni-bremen.de/handle/elib/405
    """
    def __init__(self,
                 ts_handle: SymbolicWeightedTransitionSystem,
                 dfa_handles: List[SymbolicAddDFA],
                 ts_curr_vars: List[ADD],
                 ts_next_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 dfa_next_vars: List[ADD],
                 ts_obs_vars: list,
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.ts_handle = ts_handle
        self.dfa_handle_list = dfa_handles
        self.init_TS = ts_handle.sym_add_init_states
        self.target_DFA_list = [dfa_tr.sym_goal_state for dfa_tr in dfa_handles]
        self.init_DFA_list = [dfa_tr.sym_init_state for dfa_tr in dfa_handles]

        self.monolithic_dfa_init = reduce(lambda x, y: x & y, self.init_DFA_list)
        self.monolithic_dfa_target = reduce(lambda x, y: x & y, self.target_DFA_list)

        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_transition_fun_list = ts_handle.sym_tr_actions

        self.dfa_transition_fun_list = [dfa_tr.dfa_bdd_tr for dfa_tr in dfa_handles] 
        self.dfa_monolithic_tr_func = reduce(lambda a, b: a & b,  self.dfa_transition_fun_list)


        self.ts_add_sym_to_curr_state_map: dict = ts_handle.predicate_add_sym_map_curr.inv
        self.ts_bdd_sym_to_curr_state_map: dict = ts_handle.predicate_sym_map_curr.inv
        self.ts_bdd_sym_to_S2obs_map: dict =  ts_handle.predicate_sym_map_lbl.inv
        self.ts_add_sym_to_S2obs_map: dict =  ts_handle.predicate_add_sym_map_lbl.inv
        self.dfa_bdd_sym_to_curr_state_map: List[dict] = [i.dfa_predicate_sym_map_curr.inv for i in dfa_handles]
        self.dfa_add_sym_to_curr_state_map: List[dict] = [i.dfa_predicate_add_sym_map_curr.inv for i in dfa_handles]

        self.obs_add: ADD = ts_handle.sym_add_state_labels
        self.tr_action_idx_map: dict = ts_handle.tr_action_idx_map

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
    

    def _construct_composed_tr_function(self) -> List[ADD]:
        """
        A function that construct that conjoints the TR function of the TS and DFA along with S2P (state to obs ADD).

        Note: We prime the S2P ADD because we want to extract the next state in the DFA after we evolve over the TS.
        """

        obs_bdd_prime = self.obs_add.swapVariables(self.ts_x_list, self.ts_y_list) 
        composed_tr_list = []
        for tr_action in self.ts_transition_fun_list:
            composed_tr = tr_action & obs_bdd_prime & self.dfa_monolithic_tr_func
            composed_tr_list.append(composed_tr)
        
        return composed_tr_list
    

    def _get_max_tr_action_cost(self) -> int:
        """
        A helper function that retireves the highest cost amongst all the transiton function costs
        """
        _max = 0
        for tr_action in self.ts_transition_fun_list:
            if not tr_action.isZero():
                action_cost = tr_action.findMax()
                action_cost_int = int(re.findall(r'\d+', action_cost.__repr__())[0])
                if action_cost_int > _max:
                    _max = action_cost_int
        
        return _max
    

    def _create_dfa_cubes(self):
        """
        A helper function that create cubses of each DFA and store them in a list in the same order as the DFA handles. These cubes are used
         when we convert a BDD to DFA state where we need to extract each DFA state.
        """
        dfa_xcube_list = [] 
        for handle in self.dfa_handle_list:
            dfa_xcube_list.append(reduce(lambda x, y: x & y, handle.sym_add_vars_curr))
        
        return dfa_xcube_list
    

    def composed_symbolic_dijkstra_nLTL(self, verbose: bool = False) -> dict:
        """
        This function implements a Dijkstra's algorithm for nLTL formulas. 

        We implement bucket based approach to expand and store the search frontier.
        """
        # compute cubes of each DFA, used only for looking up DFA state in the monolithic DFATR
        dfa_xcube_list = self._create_dfa_cubes()

        open_list = {}
        closed = self.manager.addZero()

        c_max = self._get_max_tr_action_cost()
        empty_bucket_counter: int = 0
        g_val = self.manager.addZero()
        if g_val.isZero():
            g_layer = 0
        
        # add the init state to ite respective DFA state. Note, we could start in some other state than the usual T0_init
        open_list[g_layer] = self.init_TS & self.monolithic_dfa_init

        while not self.monolithic_dfa_target <= open_list[g_layer].existAbstract(self.ts_xcube):
            # remove all states that have been explored
            open_list[g_layer] = open_list[g_layer] & ~closed

            # If unexpanded states exist ... 
            if not open_list[g_layer].isZero():
                if verbose:
                    print(f"********************Layer: {g_layer }**************************")
                
                # reset the empty bucket counter 
                empty_bucket_counter = 0
                # Add states to be expanded next to already expanded states
                closed |= open_list[g_layer]

                for prod_tr_action in self.composed_tr_list:
                    if prod_tr_action.isZero():
                        continue
                    
                    # first get the corresponding transition action cost (constant at the terminal node)
                    action_cost = prod_tr_action.findMax()
                    step = g_val + action_cost
                    step_val = int(re.findall(r'\d+', step.__repr__())[0])
                

                    # compute the image of the TS states 
                    image_prod_add = self.image_per_action(trans_action=prod_tr_action,
                                                        From=open_list[g_layer],
                                                        xcube=self.prod_xcube,
                                                        x_list=self.prod_xlist,
                                                        y_list=self.prod_ylist)
                    
                    if image_prod_add.isZero():
                        continue
                    
                    prod_image_restricted = image_prod_add.existAbstract(self.ts_obs_cube)
                    prod_image_restricted = prod_image_restricted.bddPattern().toADD()
                    
                    if verbose:
                        self.get_prod_states_from_dd(dd_func=image_prod_add,
                                                     obs_flag=False,
                                                     dfa_xcube_list=dfa_xcube_list)
                
                    # if the bucket exists then take the union else initialize the bucket
                    if step_val in open_list:
                        open_list[step_val] |= prod_image_restricted
                    else:
                        open_list[step_val] = prod_image_restricted
            
            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == c_max:
                    print("No plan exists! Terminating algorithm.")
                    sys.exit(-1)
            

            g_val = g_val + self.manager.addOne()
            g_layer += 1

            # keep updating g_layer up until the most recent bucket
            while g_layer not in open_list:
                g_val = g_val + self.manager.addOne()
                g_layer += 1
        
        open_list[g_layer] = open_list[g_layer] & self.monolithic_dfa_target

        if verbose:
            print("********************The goal state encountered is***********************")
            self.get_prod_states_from_dd(dd_func=open_list[g_layer],
                                         obs_flag=False,
                                         dfa_xcube_list=dfa_xcube_list)
        
        print(f"Found a plan with least cost lenght {g_layer}, Now retireving it!")

        return self.retrieve_composed_symbolic__dijkstra_nLTL(max_layer=g_layer,
                                                              freach_list=open_list,
                                                              verbose=verbose)


    def retrieve_composed_symbolic__dijkstra_nLTL(self, max_layer: int, freach_list: dict, verbose: bool = False):
        """
        Retrieve the plan through Backward search by starting from the Goal state and computing the interseaction of Forwards and Backwards
         Reachable set. 
        """
        # compute cubes of each DFA, used only for looking up DFA state in the monolithic DFATR
        dfa_xcube_list = self._create_dfa_cubes()

        g_layer = self.manager.addConst(int(max_layer))
        g_int = int(re.findall(r'-?\d+', g_layer.__repr__())[0])
        print("Working Retrieval plan now")

        current_prod = freach_list[g_int]
        composed_prod_state = self.init_TS & self.monolithic_dfa_init

        parent_plan = {}

        while not composed_prod_state <= freach_list[g_int]:
            # new_current_prod = self.manager.addZero()
            for tr_num, prod_tr_action in enumerate(self.composed_tr_list):
                pred_prod= self.pre_per_action(trans_action=prod_tr_action,
                                               From=current_prod,
                                               ycube=self.prod_ycube,
                                               x_list=self.prod_xlist,
                                               y_list=self.prod_ylist)
                
                if pred_prod.isZero():
                    continue
                    
                # first get the corresponding transition action cost (constant at the terminal node)
                action_cost_cnst = prod_tr_action.findMax()
                step = g_layer - action_cost_cnst
                if step.isZero():
                    step_val = 0
                else:
                    step_val = int(re.findall(r'-?\d+', step.__repr__())[0])
                    # there can be cases where step_val cam go negtive, we skip such iterations
                    if step_val < 0:
                        continue
                    
                if pred_prod & freach_list.get(step_val, self.manager.addZero()) != self.manager.addZero():
                    # store the predecessor per action
                    tmp_current_prod = pred_prod & freach_list[step_val]
                    
                    if verbose:
                        self.get_prod_states_from_dd(dd_func=tmp_current_prod, obs_flag=False, dfa_xcube_list=dfa_xcube_list)
                    
                    tmp_current_prod_res = tmp_current_prod.existAbstract(self.ts_obs_cube)
                    
                    self._append_dict_value_composed(parent_plan,
                                                     key_prod=tmp_current_prod_res,
                                                     action=self.tr_action_idx_map.inv[tr_num])
                    
                    current_prod = tmp_current_prod_res 

                    g_layer = step 
                    break

            # current_prod = new_current_prod

            if g_layer.isZero():
                g_int = 0
            else:
                g_int = int(re.findall(r'-?\d+', g_layer.__repr__())[0])
            assert  g_int >= 0, "Error Retrieving a plan. FIX THIS!!"

            if verbose:
                print(f"********************Layer: {g_int}**************************")
                self.get_prod_states_from_dd(dd_func=tmp_current_prod, obs_flag=False,  dfa_xcube_list=dfa_xcube_list)
            
        
        return parent_plan

