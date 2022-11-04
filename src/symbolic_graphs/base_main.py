'''
Base class used by different problem instances like gridworld, frankworld to construct a symbolic TS, DFA(s)
'''
import math

from cudd import Cudd, BDD, ADD
from typing import Tuple, List, Dict, Union

from src.explicit_graphs import TwoPlayerGame
from src.explicit_graphs import Ltlf2MonaDFA

from src.symbolic_graphs import SymbolicDFA, SymbolicAddDFA
from src.symbolic_graphs import SymbolicTransitionSystem, SymbolicWeightedTransitionSystem

class BaseSymMain():

    def __init__(self,
                 domain_file: str, 
                 problem_file: str,
                 formulas: Union[List, str],
                 manager: Cudd,
                 plot_dfa: bool = False,
                 ltlf_flag: bool = True,
                 dyn_var_ord: bool = False,):
        
        self.domain_file: str = domain_file
        self.problem_file: str = problem_file

        self.formulas = formulas

        self.manager = manager

        self.plot_dfa = plot_dfa
        self.ltlf_flag: bool = ltlf_flag
        self.dyn_var_ordering: bool = dyn_var_ord
    

    def build_abstraction(self):
        """
        A method that constructs the symbolic TS and Symbolic DFA/DFAs as required for graph search or strategy synthesis. 

        Every class that inherits this base class has to implement this method
        """
        raise NotImplementedError()
    

    def build_bdd_abstraction(self):
        """
         Main Function to Build Transition System that only represent valid edges without any weights
        """
        raise NotImplementedError()
    

    def build_weighted_add_abstraction(self):
        """
         Main Function to Build Transition System that represents valid edges without their corresponding weights

         Pyperplan supports the following PDDL fragment: STRIPS without action costs
        """
        raise NotImplementedError()
    

    def create_symbolic_dfa_graph(self, 
                                  formula: str,
                                  dfa_num: int,
                                  add_flag: bool = False):
        # Construct DFA from ltlf
        if self.ltlf_flag:
            _dfa = Ltlf2MonaDFA(formula=formula)
            _num_of_states = _dfa.num_of_states
            
        # Construct DFA from ltl using SPOT
        else:
            _two_player_instance = TwoPlayerGame(None, None)
            _dfa = _two_player_instance.build_LTL_automaton(formula=formula, plot=False)
            _state = _dfa.get_states()
            _num_of_states = len(_state)

        # the number of boolean variables (|a|) = log⌈|DFA states|⌉
        curr_state, next_state = self.create_symbolic_vars(num_of_facts=_num_of_states,
                                                           curr_state_var_name=f'a{dfa_num}_',
                                                           next_state_var_name=f'b{dfa_num}_',
                                                           add_flag=add_flag)
        return curr_state, next_state, _dfa
    

    def create_symbolic_vars(self,
                             num_of_facts: int,
                             curr_state_var_name: str = 'x',
                             next_state_var_name: str = 'y',
                             add_flag: bool = False) -> Tuple[list, list]:
        """
        A helper function to create log⌈num_of_facts⌉ boolean variables. 

        If the ADD flag is set to True, then create ADD Variables else create BDD Variables. 
        """
        curr_state_vars: list = []
        next_state_vars: list = []

        cur_state = curr_state_var_name
        nxt_state = next_state_var_name

        # get the number of variables in the manager. We will assign the next idex to the next lbl variables
        _num_of_sym_vars = self.manager.size()

        for num_var in range(math.ceil(math.log2(num_of_facts))):
            if add_flag:
                curr_state_vars.append(self.manager.addVar(_num_of_sym_vars + (2*num_var), f'{cur_state}{num_var}'))
                next_state_vars.append(self.manager.addVar(_num_of_sym_vars + (2*num_var + 1), f'{nxt_state}{num_var}'))
            else:
                curr_state_vars.append(self.manager.bddVar(_num_of_sym_vars + (2*num_var), f'{cur_state}{num_var}'))
                next_state_vars.append(self.manager.bddVar(_num_of_sym_vars + (2*num_var + 1), f'{nxt_state}{num_var}'))

        return (curr_state_vars, next_state_vars)



    def build_bdd_symbolic_dfa(self,  sym_tr_handle: SymbolicTransitionSystem)  -> Tuple[List[SymbolicDFA], List[BDD], List[BDD]]:
        """
        A helper function to build a symbolic DFA given a formul from BDD Variables.
        """

        # create a list of DFAs
        DFA_handles = []
        DFA_curr_vars = []
        DFA_nxt_vars = []

        for _idx, fmla in enumerate(self.formulas):
            # create different boolean variables for different DFAs - [ai_0 for ith DFA]
            dfa_curr_state, dfa_next_state, _dfa = self.create_symbolic_dfa_graph(formula= fmla,
                                                                                  dfa_num=_idx)

            # create TR corresponding to each DFA - dfa name is only used dumping graph 
            dfa_tr = SymbolicDFA(curr_states=dfa_curr_state,
                                 next_states=dfa_next_state,
                                 predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                 manager=self.manager,
                                 dfa=_dfa,
                                 ltlf_flag=self.ltlf_flag,
                                 dfa_name=f'dfa_{_idx}')
            if self.ltlf_flag:
                dfa_tr.create_symbolic_ltlf_transition_system(verbose=self.verbose, plot=self.plot_dfa)
            else:
                dfa_tr.create_dfa_transition_system(verbose=self.verbose,
                                                    plot=self.plot_dfa,
                                                    valid_dfa_edge_formula_size=len(_dfa.get_symbols()))

            # We extend DFA vars list as we dont need them stored in separate lists
            DFA_handles.append(dfa_tr)
            DFA_curr_vars.extend(dfa_curr_state)
            DFA_nxt_vars.extend(dfa_next_state)
        
        return DFA_handles, DFA_curr_vars, DFA_nxt_vars
    

    def build_add_symbolic_dfa(self, sym_tr_handle: SymbolicWeightedTransitionSystem) -> Tuple[List[SymbolicAddDFA], List[ADD], List[ADD]]:
        """
        A helper function to build a symbolic DFA given a formula from ADD Variables.
        """      
        # create a list of DFAs
        DFA_handles = []
        DFA_curr_vars = []
        DFA_nxt_vars = []

        for _idx, fmla in enumerate(self.formulas):
            # create different ADD variables for different DFAs
            add_dfa_curr_state, add_dfa_next_state, _dfa = self.create_symbolic_dfa_graph(formula=fmla,
                                                                                          dfa_num=_idx,
                                                                                          add_flag=True)

            # create TR corresponding to each DFA - dfa name is only used dumping graph 
            dfa_tr = SymbolicAddDFA(curr_states=add_dfa_curr_state,
                                    next_states=add_dfa_next_state,
                                    predicate_add_sym_map_lbl=sym_tr_handle.predicate_add_sym_map_lbl,
                                    predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                    manager=self.manager,
                                    dfa=_dfa,
                                    ltlf_flag=self.ltlf_flag,
                                    dfa_name=f'dfa_{_idx}')
            
            if self.ltlf_flag:
                dfa_tr.create_symbolic_ltlf_transition_system(verbose=self.verbose, plot=self.plot_dfa)
            else:
                dfa_tr.create_dfa_transition_system(verbose=self.verbose,
                                                    plot=self.plot_dfa,
                                                    valid_dfa_edge_formula_size=len(_dfa.get_symbols()))

            # We extend DFA vars list as we dont need them stored in separate lists
            DFA_handles.append(dfa_tr)
            DFA_curr_vars.extend(add_dfa_curr_state)
            DFA_nxt_vars.extend(add_dfa_next_state)
        
        return DFA_handles, DFA_curr_vars, DFA_nxt_vars

