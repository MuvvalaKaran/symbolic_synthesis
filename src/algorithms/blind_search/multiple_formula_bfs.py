
from cudd import Cudd, BDD, ADD
from src.algorithms.base import BaseSymbolicSearch

class MultipleFormulaBFS(BaseSymbolicSearch):
    """
    Given a Transition systenm, and n DFAs associated with different Formulas, this class computed the shorted path (in terms of # of edges taken to)
    """

    def __init__(self,
                 init_TS: BDD,
                 target_DFA: BDD,
                 init_DFA: BDD,
                 ts_curr_vars: list,
                 ts_next_vars: list,
                 dfa_curr_vars: list,
                 dfa_next_vars: list,
                 ts_obs_vars: list,
                 cudd_manager: Cudd):
        super().__init__(init_TS, target_DFA, init_DFA, ts_curr_vars, ts_next_vars, dfa_curr_vars, dfa_next_vars, ts_obs_vars, cudd_manager)
        pass