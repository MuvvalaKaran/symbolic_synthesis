import re
import sys

from functools import reduce
from typing import List, Union, Tuple

from cudd import Cudd, BDD, ADD

from src.algorithms.base import BaseSymbolicSearch
from src.algorithms.weighted_search.symbolic_BDDAstar import SymbolicBDDAStar
from src.symbolic_graphs import SymbolicMultipleDFA, SymbolicWeightedTransitionSystem


class MultipleFormulaBDDAstar(BaseSymbolicSearch):
    """
    Given a Transition systenm, and n DFAs associated with different Formulas, this class computes the minimum cost path
    by searching over the composed graph using the BDDA* algorithm.

    Algorithm inspired from Peter Kissmann's PhD thesis on - Symbolic Search in Planning and General Game Playing.
     Link - https://media.suub.uni-bremen.de/handle/elib/405
    """

    def __init__(self,
                 ts_handle: SymbolicWeightedTransitionSystem,
                 dfa_handles: SymbolicMultipleDFA,
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

        # composed graph consists of state S, Z and hence are function of TS and DFA bdd vars
        self.prod_xlist = self.ts_x_list + self.dfa_x_list
        self.prod_ylist = self.ts_y_list + self.dfa_y_list
        self.prod_xcube = reduce(lambda x, y: x & y, self.prod_xlist)
        self.prod_ycube = reduce(lambda x, y: x & y, self.prod_ylist)

        # composed monolithic TR
        self.composed_tr_list = self._construct_composed_tr_function()

        # compute indv. product state h values
        self.estimate_list, self.estimate_max = self._compute_heurstic_functions(verbose=True)

    

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
    

    def _compute_heurstic_functions(self, verbose: bool = False) -> Tuple[ADD, int]:
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

            astar_handle = SymbolicBDDAStar(ts_handle=self.ts_handle,
                                            dfa_handle=dfa_handle,
                                            ts_curr_vars=self.ts_x_list,
                                            ts_next_vars=self.ts_y_list,
                                            dfa_curr_vars=_dfa_curr_vars,
                                            dfa_next_vars=_dfa_next_vars,
                                            ts_obs_vars=self.ts_obs_list,
                                            cudd_manager=self.manager,
                                            verbose=False,
                                            print_h_vals=True)

            if astar_handle.heur_max > comp_max_heur:
                comp_max_heur = astar_handle.heur_max

            estimates_add.append(astar_handle.heur_add) 

            dfa_idx_start = dfa_idx_stop
        
        assert dfa_idx_stop == len(self.dfa_x_list) - 1, "Error while  extracting DFA vars. FIX THIS!!!"

        return estimates_add, comp_max_heur


    def composed_symbolic_Astar_search_nLTL(self, verbose: bool = False) -> dict:
        """
        This function implements a BDDA* algorithm as outlined in Peter Kissmann Ph.D. Thesis. 
        """
        raise NotImplementedError()


