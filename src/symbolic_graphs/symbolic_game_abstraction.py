import re
import sys
import copy
import graphviz as gv

from typing import Tuple, List, Dict
from cudd import Cudd, BDD, ADD

from src.symbolic_graphs.symbolic_abstraction import PartitionedFrankaTransitionSystem

class DynamicFrankaTransitionSystem(PartitionedFrankaTransitionSystem):
    """
     A class that constructs symbolc Two-player Transition Relation. 
    """

    def __init__(self,
                 curr_vars: list,
                 lbl_vars: list,
                 action_vars: list,
                 task, domain,
                 ts_state_map: dict,
                 ts_states: list,
                 manager: Cudd):
        super().__init__(curr_vars, lbl_vars, action_vars, task, domain, ts_state_map, ts_states, manager)

    
    def _initialize_bdds_for_actions(self):
        raise NotImplementedError()
    

    def add_edge_to_action_tr(self, state_start_idx: int, action_name: str, curr_state_tuple: tuple, next_state_tuple: tuple) -> None:
        raise NotImplementedError()
    

    def create_transition_system_franka(self, boxes: List[str], state_lbls: List, add_exist_constr: bool = True, verbose: bool = False, plot: bool = False):
        raise NotImplementedError()