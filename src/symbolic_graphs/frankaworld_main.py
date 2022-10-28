import sys
import math
import warnings
import copy

from typing import Tuple, List, Dict, Union
from cudd import Cudd, BDD, ADD

from src.explicit_graphs import CausalGraph, FiniteTransitionSystem

from src.symbolic_graphs import SymbolicDFA, SymbolicAddDFA
from src.symbolic_graphs import SymbolicTransitionSystem, SymbolicWeightedTransitionSystem, SymbolicFrankaTransitionSystem

from .base_main import BaseSymMain

from utls import *


class FrankaWorld(BaseSymMain):

    def __init__(self,
                 domain_file: str, 
                 problem_file: str,
                 formulas: Union[List, str],
                 manager: Cudd,
                 weight_dict: dict = {},
                 ltlf_flag: bool = True,
                 dyn_var_ord: bool = False,
                 verbose: bool = False,
                 plot_ts: bool = False,
                 plot_obs: bool = False,
                 plot_dfa: bool = False,
                 plot: bool = False,
                 create_lbls: bool = True):
        super().__init__(domain_file, problem_file, formulas, manager, plot_dfa, ltlf_flag, dyn_var_ord)

        self.weight_dict: Dict[str, int] = weight_dict

        self.verbose: bool = verbose
        self.plot: bool = plot
        self.plot_ts: bool = plot_ts
        self.plot_obs: bool = plot_obs
        self.dyn_var_ordering: bool = dyn_var_ord

        self.create_lbls: bool = create_lbls
    

    def build_abstraction(self, draw_causal_graph: bool = False):
        """
        A main function that construct a symbolic Franka World TS and its corresponsing DFA
        """
        print("*****************Creating Boolean variables for Frankaworld!*****************")

        sym_tr, ts_curr_state, ts_next_state, ts_lbl_states = self.build_bdd_abstraction(draw_causal_graph=draw_causal_graph)

        dfa_tr, dfa_curr_state, dfa_next_state = self.build_bdd_symbolic_dfa(sym_tr_handle=sym_tr)
    

    def _create_symbolic_lbl_vars(self, domain_facts, state_var_name: str, add_flag: bool = False) -> List[Union[BDD, ADD]]:
        """
         A function that create only one set of vars for the objects passed. This function does to create prime varibables. 
        """
        state_lbl_vars: list = []
        _num_of_sym_vars = self.manager.size()
        num: int = math.ceil(math.log2(len(domain_facts)))

        # happens when number of domain_facts passed as argument is 1
        if num == 0:
            num += 1
        
        for num_var in range(num):
            _var_index = num_var + _num_of_sym_vars
            if add_flag:
                state_lbl_vars.append(self.manager.addVar(_var_index, f'{state_var_name}{num_var}'))
            else:
                state_lbl_vars.append(self.manager.bddVar(_var_index, f'{state_var_name}{num_var}'))
        
        return state_lbl_vars
    

    def _segregate_predicates(self, predicates: List[str]) -> dict:
        """
        A helper function that segretaes the predicates as required by the symbolic transition relation. We separate them based on

         1) all gripper predicates - we do not need to create prime version for these
         2) all on predicates - we do need to create prime version for these
         3) all holding predicates 
         4) rest of the predicates - holding, ready, to-obj, and to-loc predicates. We do create prime version for these. 
        """

        predicate_dict = {
            'on': [],
            'others': []
        }

        for pred in predicates:
            if 'on' in pred or 'gripper' in pred:
                predicate_dict['on'].append(pred)
            else:
                predicate_dict['others'].append(pred)
        
        return predicate_dict


    def create_symbolic_causal_graph(self, draw_causal_graph: bool = False, remove_flag: bool = False, add_flag: bool = False):
        """
        A function to create an instance of causal graph which call pyperplan. We access the task related properties pyperplan
        and create symbolic TR related to action.   

        _causal_graph_instance.task.facts: Grounded facts about the world 
        _causal_graph_instance.task.initial_sttates: initial condition(s) of the world
        _causal_graph_instance.task.goals:  Desired Final condition(s) of the world
        _causal_graph_instance.task.operators: Actions that the agent (Franka) can take from all the grounded facts

        Pyperplan: Not does not natively support equality - needs to remove action like Transit b# l1 l1. More info:

         Equality keyword Github issue - https://github.com/aibasel/pyperplan/issues/13
         Eqaulity Keyword PR - https://github.com/aibasel/pyperplan/pull/15

        """
        _causal_graph_instance = CausalGraph(problem_file=self.problem_file,
                                             domain_file=self.domain_file,
                                             draw=draw_causal_graph)

        _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

        task_facts = _causal_graph_instance.task.facts
        boxes = _causal_graph_instance.task_objects

        # segregate grounded predicates into thre categories, 1) gripper predicates, 2) on predicates, 3) all other prdicates
        seg_preds = self._segregate_predicates(predicates=task_facts)

        if remove_flag:
            # if transit action, remove redundant transit to same loc
            finite_ts = FiniteTransitionSystem(_causal_graph_instance)
            # deepcopy as list is a mutuable object
            tmp_opr_list = copy.deepcopy(_causal_graph_instance.task.operators)
            for action in _causal_graph_instance.task.operators:
                if 'transit' in action.name and 'else' not in action.name:
                    _, locs = finite_ts._get_multiple_box_location(multiple_box_location_str=action.name)
                    if locs[0] == locs[1]:
                        tmp_opr_list.remove(action)
            
            # deleting from an interable that we are interating over is a bad practice and is not foolproof
            _causal_graph_instance.task.operators = tmp_opr_list

        sym_vars = dict(seg_preds)  # shallow copy to avoid changing the org content

        seg_preds['curr_state'] = seg_preds['others']
        seg_preds['next_state'] = seg_preds['others']

        del seg_preds['others']

        sym_vars['on'] = self._create_symbolic_lbl_vars(domain_facts=seg_preds['on'], state_var_name='b', add_flag=add_flag)
       

        curr_state, next_state = self.create_symbolic_vars(num_of_facts=len(sym_vars['others']),
                                                           add_flag=add_flag)

        sym_vars['curr_state'] = curr_state
        sym_vars['next_state'] = next_state
        
        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, sym_vars, seg_preds, boxes
        

    def build_bdd_abstraction(self, draw_causal_graph: bool = False):
        """
         Main Function to Build Transition System that only represent valid edges without any weights
        """
        task, domain, ts_sym_vars, seg_preds, boxes = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                 remove_flag=True)

        sym_tr = SymbolicFrankaTransitionSystem(sym_vars_dict=ts_sym_vars,
                                                task=task,
                                                domain=domain,
                                                boxes=boxes,
                                                manager=self.manager,
                                                seg_facts=seg_preds) 

        sym_tr.create_transition_system_franka(add_exist_constr=True,
                                               verbose=True,
                                               plot=self.plot_ts)

        return sym_tr, ts_curr_state, ts_next_state, None


    def build_weighted_add_abstraction(self):
        pass