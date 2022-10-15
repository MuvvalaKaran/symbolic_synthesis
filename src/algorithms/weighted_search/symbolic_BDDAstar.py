import re
import sys

from functools import reduce
from tabnanny import verbose

from cudd import Cudd, BDD, ADD
from typing import Union, List
from config import GRID_WORLD_SIZE

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import SymbolicAddDFA, SymbolicWeightedTransitionSystem



class SymbolicBDDAStar(BaseSymbolicSearch):

    def __init__(self,
                 ts_handle: SymbolicWeightedTransitionSystem,
                 dfa_handle: SymbolicAddDFA,
                 ts_curr_vars: List[ADD],
                 ts_next_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 dfa_next_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = ts_handle.sym_add_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_transition_fun_list = ts_handle.sym_tr_actions
        self.dfa_transition_fun = dfa_handle.dfa_bdd_tr
        self.ts_add_sym_to_curr_state_map: dict = ts_handle.predicate_add_sym_map_curr.inv
        self.ts_bdd_sym_to_curr_state_map: dict = ts_handle.predicate_sym_map_curr.inv
        self.ts_bdd_sym_to_S2obs_map: dict = ts_handle.predicate_sym_map_lbl.inv
        self.ts_add_sym_to_S2obs_map: dict = ts_handle.predicate_add_sym_map_lbl.inv
        self.dfa_bdd_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_sym_map_curr.inv
        self.dfa_add_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_add_sym_map_curr.inv
        self.obs_add = ts_handle.sym_add_state_labels
        self.tr_action_idx_map = ts_handle.tr_action_idx_map

        # create corresponding cubes to avoid repetition
        self.ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        self.ts_ycube = reduce(lambda x, y: x & y, self.ts_y_list)
        self.ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)

        self.dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)

        # composed graph consists of state S, Z and hence are function TS and DFA vars
        self.prod_xlist = self.ts_x_list + self.dfa_x_list
        self.prod_ylist = self.ts_y_list + self.dfa_y_list
        self.prod_xcube = reduce(lambda x, y: x & y, self.prod_xlist)
        self.prod_ycube = reduce(lambda x, y: x & y, self.prod_ylist)

        # composed monolithic TR
        self.composed_tr_list = self._construct_composed_tr_function()

        # compute all the valid states in the Transition System
        self.ts_states: ADD = self._compute_set_of_TS(sanity_check=True)
        self.heur_add: ADD = self._compute_min_cost_to_goal(verbose=True)
    

    def _construct_composed_tr_function(self) -> List[ADD]:
        """
        A function that construct that conjoints the TR function of the TS and DFA along with S2P (state to obs BDD).

        Note: We prime the S2P BDD because we want to extract the next state in the DFA after we evolve over the TS.
        """

        obs_bdd_prime = self.obs_add.swapVariables(self.ts_x_list, self.ts_y_list) 
        composed_tr_list = []
        for tr_action in self.ts_transition_fun_list:
            composed_tr = tr_action & obs_bdd_prime & self.dfa_transition_fun
            composed_tr_list.append(composed_tr)
        
        return composed_tr_list
    

    def _compute_set_of_TS(self, sanity_check: bool = False) -> ADD:
        """
        A function commpute the set of all Transition System states
        """
        ts_states: BDD = self.manager.bddZero()
        for tr_ac in self.ts_transition_fun_list:
            # ts_states |= tr_ac.existAbstract(self.ts_ycube.bddPattern()).bddPattern() 
            ts_states |= tr_ac.bddPattern().existAbstract(self.ts_ycube.bddPattern())

        ts_x_list_bdd = [var.bddPattern() for var in self.ts_x_list]

        if sanity_check:
            cubes = self.convert_cube_to_func(dd_func=ts_states, curr_state_list=ts_x_list_bdd)
            assert len(cubes) == GRID_WORLD_SIZE**2, "Error computing set of valid TS states"
            assert self.init_TS.bddPattern() <= ts_states, "Error computing set of valid TS states"

        return ts_states.toADD()
    

    def _get_max_tr_action_cost(self) -> int:
        """
        A helper function that retireves the highest cost amongst all the transiton function costs
        """
        _max = 0
        for tr_action in self.ts_transition_fun_list:
                action_cost = tr_action.findMax()
                action_cost_int = int(re.findall(r'\d+', action_cost.__repr__())[0])
                if action_cost_int > _max:
                    _max = action_cost_int
        
        return _max
    
    def _convert_vector_BDD_to_ADD(self, reach_list: List[ADD], verbose: bool = False) -> ADD:
        """
        A function that compute the minimum distance to the accepting state in a product automaton given a vectors of ADDs. 

        Implementation: Starting from the first "bucket" in the vectors of ADDs, we assign all the state (s, z_acc), where z_acc \in Z_acc 
        is an accepting DFA state, a h value of zero (becuse they are already accepting states). Then, we iterate through the rest of vector of
        ADDs and store their min cost-to-goal. 
        """
        closed_add: ADD = self.manager.plusInfinity()

        for layer, bucket in reach_list.items():
            # if the bucket has accepting states, assign them a h value of zero and add them to the closed ADD
            accp_states_og: ADD = bucket.restrict(self.target_DFA)

            # if accetpgin states exists. . .
            if not accp_states_og.isZero():
                # add the DFA accepting state
                prod_accp_s: ADD = accp_states_og & self.target_DFA
                # sanity checking 
                assert prod_accp_s <= bucket, "Error computing the h value from the accepting product states"

                prod_accp_states: ADD = prod_accp_s.ite(self.manager.addZero(), self.manager.plusInfinity())
                closed_add = closed_add.min(prod_accp_states)

                # remove the accp_state that have already been added
                new_bucket: ADD = bucket - prod_accp_s

            else:
                new_bucket: ADD = bucket

            # add the minimum cost to goal
            if not new_bucket.isZero():
                # assign the current value to the state's h value and then add them to ADD
                new_bucket_w = new_bucket.ite(self.manager.addConst(int(layer)), self.manager.plusInfinity())

                
                closed_add = closed_add.min(new_bucket_w)
        
        if verbose:
            for layer in reach_list.keys():
                prod_states = closed_add.bddInterval(layer, layer).toADD()
                if not prod_states.isZero():
                    # convert to bdd and print out the prod state h value
                    print(f"*****************States with h value {layer}****************************")
                    self.get_prod_states_from_dd(dd_func=prod_states, obs_flag=False)

        return closed_add
    

    def _compute_min_cost_to_goal(self, verbose: bool = False) -> ADD:
        """
        Given a product Trasition Relation (TR) corresponding to one formula, compute the h value assocated with each
         product state (s, z) where s \in S belongs to the Treansition System and z \in Z belongs to the DFA for \phi_i where
         i is the ith formula.

        Functionality: Perform backwards reachability in a dijkstras fashion and retain the minimum cost-to-goal from rach (s, z). 
        All (s, z) where z \in Z_acc (accepting DFA state) have h value of zero. 
        """

        open_list = {}
        closed = self.manager.addZero()
        c_max = self._get_max_tr_action_cost()

        composed_prod_state = self.ts_states & self.target_DFA

        # counter used for breaking
        empty_bucket_counter: int = 0

        # counter 
        g_val = self.manager.addZero()
        if g_val.isZero():
            g_layer = 0

        # add the product accepting state to the open list
        open_list[g_layer] = composed_prod_state

        if verbose:
            self.get_prod_states_from_dd(dd_func=composed_prod_state, obs_flag=False)

        # while not composed_prod_state <= freach_list[g_int]:
        while not open_list[g_layer].isZero():
            # new_current_prod = self.manager.addZero()
            # remove all states that have been explored
            open_list[g_layer] = open_list[g_layer] & ~closed

            # If unexpanded states exist ... 
            if not open_list[g_layer].isZero():
                if verbose:
                    print(f"********************Layer: {g_layer + 1}**************************")
                
                # reset the empty bucket counter 
                empty_bucket_counter = 0
                # Add states to be expanded next to already expanded states
                closed |= open_list[g_layer]

                # since this is a single formula code, we do not have to explicity create a composed to every formula as we only have one.
                for prod_tr_action in self.composed_tr_list:
                    # first get the corresponding transition action cost (constant at the terminal node)
                    action_cost = prod_tr_action.findMax()
                    step = g_val + action_cost
                    step_val = int(re.findall(r'\d+', step.__repr__())[0])

                    pred_prod= self.pre_per_action(trans_action=prod_tr_action,
                                                   From=open_list[g_layer],
                                                   ycube=self.prod_ycube,
                                                   x_list=self.prod_xlist,
                                                   y_list=self.prod_ylist)
                    
                    if pred_prod.isZero():
                        continue
                        
                    prod_image_restricted = pred_prod.existAbstract(self.ts_obs_cube)
                
                    if verbose:
                        self.get_prod_states_from_dd(dd_func=pred_prod, obs_flag=False)
                    
                    # if the bucket exists then take the union else initialize the bucket
                    if step_val in open_list:
                        open_list[step_val] |= prod_image_restricted
                    else:
                        open_list[step_val] = prod_image_restricted
            
            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == c_max:
                    print("Reached a Fix Point!")
                    break

            g_val = g_val + self.manager.addOne()
            g_layer += 1

            if g_layer > max(open_list.keys()):
                g_layer -= 1
                break

            # keep updating g_layer up until the most recent bucket
            while g_layer not in open_list:
                g_val = g_val + self.manager.addOne()
                g_layer += 1
                
        print(f"********************Took {g_layer} layers to reach a fixed point********************")

        # retain the minimum distance to goal
        estimate_add: ADD = self._convert_vector_BDD_to_ADD(reach_list=open_list, verbose=verbose)
            
        return estimate_add
    

    def composed_symbolic_Astar_search(self, verbose: bool = False):
        """
        A function that symbolic BDDA* algorithm by Prof. Stefan Edelkam and Peeter Kissmann in his Ph.D. dissertation thesis.
        """

        raise NotImplementedError()


