import re
import sys

from copy import deepcopy
from math import inf
from functools import reduce
from typing import List, Union, Tuple

from cudd import Cudd, BDD, ADD

from src.algorithms.base import BaseSymbolicSearch
from src.algorithms.weighted_search.symbolic_BDDAstar import SymbolicBDDAStar
from src.symbolic_graphs import SymbolicMultipleDFA, SymbolicWeightedTransitionSystem


class PreferenceBDDAstar(BaseSymbolicSearch):
    """
    Given a Transition systenm, and n DFAs associated with different Formulas, this class computes the minimum cost path
     that  also satisfies user's preference accordingly to Peter algorithm from ICRA23 paper using the BDDA* algorithm.

    """

    def __init__(self,
                 ts_handle: SymbolicWeightedTransitionSystem,
                 dfa_handles: SymbolicMultipleDFA,
                 ts_curr_vars: List[ADD],
                 ts_next_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 dfa_next_vars: List[ADD],
                 ts_obs_vars: list,
                 cudd_manager: Cudd,
                 verbose: bool = False,
                 print_h_vals: bool = False):
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

        # composed graph consists of state S, Z and hence are function of TS and DFA bdd vars
        self.prod_xlist = self.ts_x_list + self.dfa_x_list
        self.prod_ylist = self.ts_y_list + self.dfa_y_list
        self.prod_xcube = reduce(lambda x, y: x & y, self.prod_xlist)
        self.prod_ycube = reduce(lambda x, y: x & y, self.prod_ylist)

        # compute cubes of each DFA, used only for looking up DFA state in the monolithic DFATR
        self.dfa_xcube_list = self._create_dfa_cubes()

        # composed monolithic TR
        self.composed_tr_list = self._construct_composed_tr_function()

        # compute indv. product state h values
        self.estimate_list, self.estimate_max = self._compute_heurstic_functions(verbose=verbose, print_h_vals=print_h_vals)


    def _create_dfa_cubes(self):
        """
        A helper function that create cubses of each DFA and store them in a list in the same order as the DFA handles. These cubes are used
         when we convert a BDD to DFA state where we need to extract each DFA state.
        """
        dfa_xcube_list = [] 
        for handle in self.dfa_handle_list:
            dfa_xcube_list.append(reduce(lambda x, y: x & y, handle.sym_add_vars_curr))
        
        return dfa_xcube_list


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
    

    def _compute_heurstic_functions(self, verbose: bool = False, print_h_vals: bool = False) -> Tuple[ADD, int]:
        """
        A function that compute the heursitic value of each product states associated with each DFA in the list of formulas.

        Implementation: We first extract the corresponding  DFA vars as there all stored in one list.
         DFA vars are converted to their string repr which gives us, for e.g., a0_0. Here a0 corresponding to the first
         formula as 0 corrresponds the formula index.

         We then create an instance of BDDA* class to compute the huerstic for product graph corresponding to the individual formula.
        """

        estimates_add = []
        dfa_idx_start: int = 0
        comp_max_heur: int = 0

        # loop over each forumula and compute heursitc function
        for dfa_num, dfa_handle in enumerate(self.dfa_handle_list):
            # extract the corresponding DFA ADD Vars
            for dfa_var_num,  dfa_var in enumerate(self.dfa_x_list[dfa_idx_start:]):
                if str(dfa_num) not in re.split('_', dfa_var.bddPattern().__repr__())[0]:
                    # breaks after encountering the first nonconformat DFA Var
                    dfa_idx_stop = dfa_var_num + dfa_idx_start
                    break
            
            assert dfa_idx_start <= dfa_idx_stop, "Error extracting DFA vars while computing ind. prod state h values. FIX THIS!!!"

            # this happen only when the number of vars per dfa or the very last formula consists of DFA var
            if dfa_idx_start == dfa_idx_stop:
                _dfa_curr_vars = self.dfa_x_list[dfa_idx_start:]
                _dfa_next_vars = self.dfa_y_list[dfa_idx_start:]
            else:
                # extract the corresponding DFA vars; +1 to inclue the last element as well. 
                _dfa_curr_vars = self.dfa_x_list[dfa_idx_start: dfa_idx_stop]
                _dfa_next_vars = self.dfa_y_list[dfa_idx_start: dfa_idx_stop]     

            if print_h_vals:
                print(f"**************************Heuristic For DFA {dfa_num}**************************")

            astar_handle = SymbolicBDDAStar(ts_handle=self.ts_handle,
                                            dfa_handle=dfa_handle,
                                            ts_curr_vars=self.ts_x_list,
                                            ts_next_vars=self.ts_y_list,
                                            dfa_curr_vars=_dfa_curr_vars,
                                            dfa_next_vars=_dfa_next_vars,
                                            ts_obs_vars=self.ts_obs_list,
                                            cudd_manager=self.manager,
                                            verbose=verbose,
                                            print_h_vals=print_h_vals)

            if astar_handle.heur_max > comp_max_heur:
                comp_max_heur = astar_handle.heur_max

            estimates_add.append(astar_handle.heur_add) 

            dfa_idx_start = dfa_idx_stop

        return estimates_add, comp_max_heur
    

    def _print_heuristic_add(self, state_vals: ADD) -> None:
        """
        A helper function that prints the state value computed using the monolithic heuristic function
        """
        # compute the set of heuristic vals 
        est_cubes: List[tuple] =  list(filter(lambda x: x[1] != inf, list(state_vals.generate_cubes())))
        
        for _, layer in est_cubes:
            prod_states = state_vals.bddInterval(layer, layer).toADD()
            if not prod_states.isZero():
                print(f"*****************States with h value {layer}****************************")
                self.get_prod_states_from_dd(dd_func=prod_states, obs_flag=False, dfa_xcube_list=self.dfa_xcube_list)


    def _get_state_estimate(self, dd_func: ADD, verbose: bool = False) -> ADD:
        """
        A helper function that extract the heurstic associated with every state in dd_func.

        The heurstis computed as max(h(s, zi) for all i) here h(s, zi) is the heuristic value
         compute for  each individual formula (\phi_i). We then take the max of all the heurstic. 
        """
        # loop over all the heuristics
        state_vals: ADD = self.manager.addZero()

        for heur_add in self.estimate_list:
            # extract the values 
            tmp_state_vals = heur_add & dd_func
            state_vals = state_vals.max(tmp_state_vals)

        # sanity check
        if not state_vals.isZero():
            check_val: int = int(list(state_vals.generate_cubes())[0][1])
            assert check_val <= self.estimate_max, "Error while computing the heuristic for monolithic prod state. FIXTHIS!!!"

        if verbose:
            self._print_heuristic_add(state_vals=state_vals)
        
        return state_vals
    

    def _compute_preference_value(curr_pcv_vec: Tuple[int], accp_idx: List[int], mu_max: int) -> Tuple[int]:
        """
        A helper function that compute the new PCV vector associated with each state, sort it and check if it below the 
        """


    # def __preference_function(self,
    #                           mu_max: int,
    #                           state_vals: ADD,
    #                           g_val: int, action_c: int,
    #                           f_max: int, open_list: dict,
    #                           pcv_open: dict, curr_pcv_vec: Tuple[int]) -> int:
    #     """
    #     A function that compute the preference cost vector for each state in the ADD,
    #      then compute the preference value and either prune or it keep it. 
    #     """
    #     def __indices(lst, item):
    #         return [i for i, x in enumerate(lst) if x == item]

    #     # create the pcv vector 
    #     max_val = max(curr_pcv_vec)
    #     # get the index associated with those max elements
    #     idxs = __indices(curr_pcv_vec, max_val)

    #     for literal_list, tmp_h_val in list(state_vals.generate_cubes()):
    #         cube = self.manager.fromLiteralList(literal_list).toADD()  # convert cube to 0-1 ADD
    #         # if there exists no accepting state in the product state
    #         preference_accp_flag: bool = False
    #         accp_idx_list = []
    #         for accp_idx, target_dfa in enumerate(self.target_DFA_list):
    #             if not cube.restrict(target_dfa).isZero():
    #                 # there exisits atleast one accepting state
    #                 preference_accp_flag = True

    #                 # keep track of their indices.
    #                 accp_idx_list.append(accp_idx)
            
                
    #         if preference_accp_flag is False:    
    #             inttmp_h_val = 0
    #             pcv_open[cube.__hash__()] = (0, 0)
    #         else:
    #             for accp_pcv_idx in accp_idx_list:
    #                 new_pcv_idx = deepcopy(curr_pcv_vec)
    #                 if 
    #                 new_pcv_idx[accp_pcv_idx] = g_val
                

    #             inttmp_h_val = int(tmp_h_val)
    #         if g_val + action_c in open_list:
    #             if inttmp_h_val in open_list[g_val + action_c]:
    #                 open_list[g_val + action_c][inttmp_h_val] |= cube
    #             else:
    #                 open_list[g_val + action_c].update({inttmp_h_val : cube})
    #         else:
    #             open_list[g_val + action_c] = {inttmp_h_val : cube}


    #         # Update maximal f value
    #         if g_val + action_c + inttmp_h_val > f_max:
    #             f_max = g_val + action_c + inttmp_h_val

    #     return f_max

            

    


    def __preference_add_state_to_ind_buckets(self,
                                              state_vals: ADD,
                                              g_val: int, action_c: int,
                                              f_max: int, open_list: dict,
                                              pcv_open: dict, pref_accp_flag: bool = False, 
                                              accp_flag: bool = False) -> int:
        """
        A helper called by the _add_states_to_buckets() to identify the right bucket and add the states to it.  

        If an accepting state BDD is passed then, manually override the associoated cube's state value to zero and add
         it to its corresponding vucket
        """

        # assert that when the accpeting flag is True, the state_val dd only has accepting prod states in it.
        if accp_flag:
            assert state_vals.restrict(~self.monolithic_dfa_target).isZero() is True, "Error Adding the accepting states to its respective bucket."

        for literal_list, tmp_h_val in list(state_vals.generate_cubes()):
            cube = self.manager.fromLiteralList(literal_list).toADD()  # convert cube to 0-1 ADD
            #  if there exists no accepting state in the state
            if pref_accp_flag is False: # all cubes have (0, 0) pcv value
                # if cube.__hash__() in pcv_open.keys():
                pcv_open[cube.__hash__()] = (0, 0)

            if not accp_flag:
                inttmp_h_val = int(tmp_h_val)
            else:
                inttmp_h_val = 0

            if g_val + action_c in open_list:
                if inttmp_h_val in open_list[g_val + action_c]:
                    open_list[g_val + action_c][inttmp_h_val] |= cube
                else:
                    open_list[g_val + action_c].update({inttmp_h_val : cube})
            else:
                open_list[g_val + action_c] = {inttmp_h_val : cube}


            # Update maximal f value
            if g_val + action_c + inttmp_h_val > f_max:
                f_max = g_val + action_c + inttmp_h_val
        
        return f_max
    

    def _preference_add_states_to_bucket(self,
                                         prod_image: ADD,
                                         g_val: int, action_c: int,
                                         f_max: int, open_list: dict,
                                         pcv_open: dict) -> int:
        """
        A helper function that s used to compute the state's h value and add it to the bucket.
        """
        # Note: ADD `&` operation implies product. Since Image return 0-1 ADD, the `&` projects the state and its corresponding h value
        # get their corresponding h values 
        # if accepting states exists. . .
        # check if the current states have atleast one accepting state
        preference_accp_flag: bool = False
        accp_idx_list = []
        for accp_idx, target_dfa in enumerate(self.target_DFA_list):
            if not prod_image.restrict(target_dfa).isZero():
                preference_accp_flag = True
                accp_idx_list.append(accp_idx)

        # if none of the tasks have been satisfied then do the normal computation
        if preference_accp_flag is False:
            if not prod_image.restrict(self.monolithic_dfa_target).isZero():
                accp_states = prod_image.restrict(self.monolithic_dfa_target)
                accp_state_vals = accp_states & self.monolithic_dfa_target
                f_max = self.__preference_add_state_to_ind_buckets(state_vals=accp_state_vals,
                                                                   g_val=g_val,
                                                                   action_c=action_c,
                                                                   f_max=f_max,
                                                                   open_list=open_list,
                                                                   pcv_open=pcv_open,
                                                                   pref_accp_flag=False,
                                                                   accp_flag=True)

            state_vals = self._get_state_estimate(dd_func=prod_image, verbose=False)
        
            # Check all possible h values and Insert successors into correct bucket
            if not state_vals.isZero():
                f_max = self.__preference_add_state_to_ind_buckets(state_vals=state_vals,
                                                                   g_val=g_val,
                                                                   action_c=action_c,
                                                                   f_max=f_max,
                                                                   pcv_open=pcv_open,
                                                                   pref_accp_flag=False,
                                                                   open_list=open_list)
        else:
            # add state
            raise NotImplementedError()

        return f_max
    

    
    def preference_based_symbolic_Astar_search(self, mu_max: int, verbose: bool = False) -> dict:
        """
        A function that implements Peter's Preference based A* planning algorithm using the
         herusitc function implemented in this class.
        """

        open_list = {}
        closed = {}

        # store the preference cost vector for each state as key value pair.
        pcv_open = {}
        pcv_closed = {}

        composed_init = self.init_TS & self.monolithic_dfa_init

        # initialize the init state with (0, 0) tuple
        pcv_open[composed_init.__hash__()] = (0, 0)

        # Find f value for the initial state.
        f_dd: ADD = self._get_state_estimate(dd_func=composed_init, verbose=verbose)
        f_val: ADD = f_dd.restrict(composed_init)

        assert f_val.isConstant() is True, "Error computing F value for the Initial prod state while initializing A* search algorithm"

        # get the int value
        f_val: int = int(list(f_val.generate_cubes())[0][1])

        # Insert prod init state into the correct bucket
        open_list[0] = {f_val : composed_init}

        # Maximal f value initially is the same as that of prod init state
        f_max = f_val
        
        # so far, no states have been expanded
        while True:
            # Stop when all states expanded 
            if f_val > f_max:
                print("********************No Plan Found********************")
                return
            
            # follow the f diagonal
            if verbose:
                print(f"********************Expanding States with f: {f_val}********************\n")
            

            for g_val in range(f_val + 1):
                h_val = f_val - g_val  # Determine the h value

                # We cannot have h values greater than max estimated value
                if h_val > self.estimate_max:
                    continue
                
                 
                if open_list.get(g_val, {}).get(h_val) is None:
                    continue 
                
                # Remove all states already expanded with same h value
                open_list[g_val][h_val] = open_list[g_val][h_val] & ~closed.get(h_val, self.manager.addZero())

                # remove all states already expanded 
                for key, val in pcv_closed.items():
                    if (key, val) in pcv_open.items():
                        pcv_open.pop(key)

                # If current bucket not empty. . .
                if not open_list[g_val][h_val].isZero():
                    # if goal state found. . .
                    if h_val == 0 and (not open_list[g_val][h_val].restrict(self.monolithic_dfa_target).isZero()):
                        open_list[g_val][h_val] = open_list[g_val][h_val] & self.monolithic_dfa_target
                        print(f"********************Found a plan with least cost lenght {g_val}, Now retireving it!********************")
                        return self.retrieve_composed_symbolic_Astar_nLTL(g_val=g_val, freach_list=open_list, verbose=verbose)
                    
                    # Add states to be expanded next to closed list
                    if h_val in closed:
                        # if the bucket exists then take the union else initialize the bucket
                        closed[h_val] |= open_list[g_val][h_val]
                    else:
                        closed[h_val] = open_list[g_val][h_val]      
                    
                    if verbose:
                        # self.get_prod_states_from_dd(dd_func=image_prod_add, obs_flag=False)
                        print(f"********************Expanding States with g: {g_val} h:{h_val}********************")
                        self.get_prod_states_from_dd(open_list[g_val][h_val], obs_flag=False, dfa_xcube_list=self.dfa_xcube_list)
                        print("\n")
                    
                    for literal_ls, _ in list(open_list[g_val][h_val].generate_cubes()):
                        # get the current pcv value from the open bucket 
                        cube = self.manager.fromLiteralList(literal_ls).toADD()
                        hash_val: int = cube.__hash__()
                        pcv_vec = pcv_open[hash_val]
                        if (hash_val, pcv_vec) not in pcv_closed.items():
                            pcv_closed[hash_val] = pcv_vec

                        ts_image_add: ADD = self.manager.addZero()
                        # Calculate successors. . .
                        for prod_tr_action in self.composed_tr_list:
                            # first get the corresponding transition action cost (constant at the terminal node)
                            action_cost: ADD = prod_tr_action.findMax()
                            assert action_cost.isConstant() is True, "Error computing action cost during A* search algorithm"
                            intaction_cost: int = int(list(action_cost.generate_cubes())[0][1])
                            # compute the image of the TS states 
                            image_prod_add: ADD = self.image_per_action(trans_action=prod_tr_action,
                                                                        From=cube,
                                                                        xcube=self.prod_xcube,
                                                                        x_list=self.prod_xlist,
                                                                        y_list=self.prod_ylist)
                            
                            ts_image_add |= image_prod_add

                        
                            if ts_image_add.isZero():
                                continue
                                
                            prod_image_restricted: ADD = ts_image_add.existAbstract(self.ts_obs_cube)

                            # if verbose:
                            #     self.get_prod_states_from_dd(dd_func=image_prod_add, obs_flag=False, dfa_xcube_list=self.dfa_xcube_list)
                            #     print(f"********************Expanding States with g: {g_val} h:{h_val}********************")
                            

                            f_max = self._preference_add_states_to_bucket(prod_image=prod_image_restricted,
                                                                          g_val=g_val,
                                                                          action_c=intaction_cost,
                                                                          f_max=f_max,
                                                                          open_list=open_list,
                                                                          pcv_open=pcv_open,
                                                                          curr_pcv_vec=pcv_vec)
                            # f_max = self.__preference_function(mu_max=mu_max,
                            #                                    state_vals=,
                            #                                    g_val=g_val,
                            #                                    action_c=intaction_cost,
                            #                                    f_max=f_max,
                            #                                    open_list=open_list,
                            #                                    pcv_open=pcv_open,
                            #                                    curr_pcv_vecpcv_vec)
        
            # Go over the next f diagonal 
            f_val += 1