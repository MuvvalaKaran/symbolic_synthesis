from multiprocessing import managers
import sys
import math
import warnings

from typing import Tuple, List, Dict, Union
from cudd import Cudd, BDD, ADD

from src.explicit_graphs import CausalGraph

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
    

    def build_abstraction(self):
        """
        A main function that construct a symbolic Franka World TS and its corresponsing DFA
        """
        print("*****************Creating Boolean variables for Frankaworld!*****************")

        sym_tr, ts_curr_state, ts_next_state, ts_lbl_states = self.build_bdd_abstraction()

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


    
    @deprecated
    def create_symbolic_lbl_vars(self, causal_graph_instance: CausalGraph,  add_flag: bool = False) -> List[Union[BDD, ADD]]:
        """ 
         A function that create symbolic variaables for the franks world:

         Based on the type od states, we need to create variables
          1) for all the boxes,
          2) for all the location, and
          3) the manipulator status (free, gripper)

        Thus, in total we need to create  log⌈|boxes|⌉ + log⌈|location|⌉ + 1 (which is log⌈|2|⌉). 

        For ease of readability, we will assigne b# for boxes, l# for location, and f# for the maniulator statues.
        """
        state_lbl_vars: list = []

        _boxes: list = causal_graph_instance.task_objects  # boxes 
        _locs: list = causal_graph_instance.task_locations  # locs

        # get the number of variables in the manager. We will assign the next idex to the next lbl variables
        _num_of_sym_vars = self.manager.size()
        num_b: int = math.ceil(math.log2(len(_boxes)))
        num_l: int = math.ceil(math.log2(len(_locs)))

        # happen when we have only one box
        if num_b == 0:
            num_b = 1

        # happen when we have only one location
        if num_l == 0:
            num_b = 1

        for num_var in range(num_b + num_l + 1):
            if num_var < num_b:
                lbl_state = 'b'
                tmp_var = num_var
            elif num_b <= num_var < num_b + num_l:
                lbl_state = 'l'
                tmp_var = num_var - num_b
            else:
                lbl_state = 'f'
                tmp_var = num_var - (num_b + num_l)
            
            _var_index = num_var + _num_of_sym_vars
            if add_flag:
                state_lbl_vars.append(self.manager.addVar(_var_index, f'{lbl_state}{tmp_var}'))
            else:
                state_lbl_vars.append(self.manager.bddVar(_var_index, f'{lbl_state}{tmp_var}'))
        
        return state_lbl_vars
    

    def _segregate_predicates(self, predicates: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        A helper function that segretaes the predicates as required by the symbolic transition relation. We separate them based on

         1) all gripper predicates - we do not need to create prime version for these
         2) all on predicates - we do need to create prime version for these
         2) rest of the predicates - holding, ready, to-obj, and to-loc predicates. We do create prime version for these. 
        """
        on_list = []
        gripper_list = []
        others_list = []
        for pred in predicates:
            if 'on' in pred:
                on_list.append(pred)
            elif 'gripper' in pred:
                gripper_list.append(pred)
            else:
                others_list.append(pred)
        
        assert len(gripper_list) == 1, "Error segregating predicates before creating sym boolean vars. FIX THIS!!!"

        return on_list, gripper_list, others_list


    def create_symbolic_causal_graph(self, draw_causal_graph: bool = False, add_flag: bool = False):
        """
        A function to create an instance of causal graph which call pyperplan. We access the task related properties pyperplan
        and create symbolic TR related to action.   

        _causal_graph_instance.task.facts: Grounded facts about the world 
        _causal_graph_instance.task.initial_sttates: initial condition(s) of the world
        _causal_graph_instance.task.goals:  Desired Final condition(s) of the world
        _causal_graph_instance.task.operators: Actions that the agent (Franka) can take from all the grounded facts

        """
        _causal_graph_instance = CausalGraph(problem_file=self.problem_file,
                                             domain_file=self.domain_file,
                                             draw=draw_causal_graph)

        _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)
        # print("No. of edges in the graph:", len(_causal_graph_instance.causal_graph._graph.edges()))

        task_facts = _causal_graph_instance.task.facts

        # segregate grounded predicates into thre categories, 1) gripper predicates, 2) on predicates, 3) all other prdicates
        on_list, gripper_list, others_list = self._segregate_predicates(predicates=task_facts)

        # for book keeping purposed
        seg_preds = {'gripper': gripper_list,
                     'on': on_list,
                     'others': others_list
                      }

        curr_state, next_state = self.create_symbolic_vars(num_of_facts=len(others_list),
                                                           add_flag=add_flag)
        
        
        # the number of boolean variables (|x|) = log⌈|facts|⌉ - Because facts represent all possible predicates in our causal graph 
        # curr_state, next_state = self.create_symbolic_vars(num_of_facts=len(task_facts),
        #                                                    add_flag=add_flag)

        gripper_var = self._create_symbolic_lbl_vars(domain_facts=gripper_list, state_var_name='f', add_flag=False)
        on_vars = self._create_symbolic_lbl_vars(domain_facts=on_list, state_var_name='b', add_flag=False)

        
        # lbl_vars = self.create_symbolic_lbl_vars(causal_graph_instance=_causal_graph_instance,
        #                                          add_flag=add_flag)
        
        # _boxes: list = _causal_graph_instance.task_objects  # boxes 
        # _locs: list = _causal_graph_instance.task_locations  # locs

        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_state, next_state, gripper_var, on_vars, seg_preds
        #  lbl_vars, _boxes, _locs
    

    def build_bdd_abstraction(self):
        """
         Main Function to Build Transition System that only represent valid edges without any weights
        """
        # task, domain, ts_curr_state, ts_next_state, ts_lbl_vars, ts_boxes, ts_locs  = self.create_symbolic_causal_graph(draw_causal_graph=False)

        task, domain, ts_curr_state, ts_next_state, ts_gripper_var, ts_on_vars, seg_preds  = self.create_symbolic_causal_graph(draw_causal_graph=False)

        sym_tr = SymbolicFrankaTransitionSystem(curr_states=ts_curr_state,
                                                next_states=ts_next_state,
                                                gripper_var=ts_gripper_var,
                                                on_vars=ts_on_vars,
                                                task=task,
                                                domain=domain,
                                                manager=self.manager,
                                                seg_facts=seg_preds) 

        sym_tr.create_transition_system_franka(task_objs=None,
                                               task_locs=None,
                                               verbose=self.verbose,
                                               plot=self.plot_ts)

        return sym_tr, ts_curr_state, ts_next_state, None


    def build_weighted_add_abstraction(self):
        pass