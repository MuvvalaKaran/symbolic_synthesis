import sys
import math
import warnings
from itertools import product
from itertools import combinations

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
    

    def __get_all_box_combos(self, boxes_dict: dict) -> List:
        """
        The franka world has the world configuration (on b# l#) embedded into it's state defination. 
        Also, we could all n but 1 boxes ground with that single box (not grounded) being currentl manipulated.

        Thus, a valid set of state labels be

        1) all boxes grounded - (on b0 l0)(on b1 l1)...
        2) all but 1 grounded - (on b0 l0)(~(on b1 l1))(on b2 l2)

        Thus, we need to create enough bool variables to accomodate all these possible configurations
        """
        parent_combo_list = []

        # create all ground configurations
        all_preds = [val for _, val in boxes_dict.items()]
        all_combos = list(product(*all_preds, repeat=1))

        parent_combo_list.extend(all_combos)

        # create n-1 combos
        num_of_boxes = len(boxes_dict)

        # create all possible n-1 combinations all boxes
        combos = combinations([*boxes_dict.keys()], num_of_boxes - 1)

        if num_of_boxes - 1 == 1:
            return parent_combo_list

        # iterate through every possible combo
        for combo in combos:
            # iterate through the tuple of boxes and create their combos
            box_loc_list = [boxes_dict[box] for box in combo]
            parent_combo_list.extend(list(product(*box_loc_list, repeat=1)))

        return parent_combo_list

    

    def _segregate_predicates(self, predicates: List[str], boxes: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        A helper function that segretaes the predicates as required by the symbolic transition relation. We separate them based on

         1) all gripper predicates - we do not need to create prime version for these
         2) all on predicates - we do need to create prime version for these
         3) all holding predicates 
         4) rest of the predicates - holding, ready, to-obj, and to-loc predicates. We do create prime version for these. 
        """

        predicate_dict = {
            # 'ready': [],
            'on': [],
            'gripper': [],
            # 'to_obj': [],
            'holding': [],
            'to_loc': [],
            'others': []
        }

        # dictionary where we segreate on predicates based on boxes - all b0, b1 ,... into seperate list 
        boxes_dict = {box: [] for box in boxes} 

        for pred in predicates:
            if 'on' in pred:
                predicate_dict['on'].append(pred)
                for b in boxes:
                    if b in pred:
                        boxes_dict[b].append(pred)
                        break
                        
            elif 'gripper' in pred:
                predicate_dict['gripper'].append(pred)
            
            else:
                predicate_dict['others'].append(pred)
                # stored separately so that we can then take =create all possible combos        
                if 'holding' in pred:
                    predicate_dict['holding'].append(pred)
                elif 'to-loc' in pred:
                    predicate_dict['to_loc'].append(pred)
        
        # we create single, n and n-1 combos - n all bozes are grounded and n-1 when one of the boxes is being manipulated
        aug_on_state = self.__get_all_box_combos(boxes_dict=boxes_dict)
        predicate_dict['on'].extend(aug_on_state)
            
        # augment the predicate list with all prermutations of holding and to-loc states 
        aug_states = list(product(predicate_dict['holding'], predicate_dict['to_loc'], repeat=1))

        # add them to the others list 
        predicate_dict['others'].extend(aug_states)
        
        assert len(predicate_dict['gripper']) == 1, "Error segregating predicates before creating sym boolean vars. FIX THIS!!!"

        # return on_list, gripper_list, holding_list, others_list
        return predicate_dict


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

        task_facts = _causal_graph_instance.task.facts
        task_boxes = _causal_graph_instance.task_objects  # boxes

        # segregate grounded predicates into thre categories, 1) gripper predicates, 2) on predicates, 3) all other prdicates
        seg_preds = self._segregate_predicates(boxes=task_boxes, predicates=task_facts)

        sym_vars = dict(seg_preds)  # shallow copy to avoid changing the org content

        # removing redundant predicates and copy predicates in others for next state look up dictionary
        del seg_preds['holding']
        del seg_preds['to_loc']

        seg_preds['curr_state'] = seg_preds['others']
        seg_preds['next_state'] = seg_preds['others']

        del seg_preds['others']

        # pass the predicates in this specific order - ready, on, gripper, to-obj, holding, to-loc
        sym_vars['on'] = self._create_symbolic_lbl_vars(domain_facts=seg_preds['on'], state_var_name='b', add_flag=add_flag)
        sym_vars['gripper'] = self._create_symbolic_lbl_vars(domain_facts=seg_preds['gripper'], state_var_name='f', add_flag=add_flag)

        curr_state, next_state = self.create_symbolic_vars(num_of_facts=len(sym_vars['others']),
                                                           add_flag=add_flag)

        sym_vars['curr_state'] = curr_state
        sym_vars['next_state'] = next_state
        
        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, sym_vars, seg_preds
        

    def build_bdd_abstraction(self, draw_causal_graph: bool = False):
        """
         Main Function to Build Transition System that only represent valid edges without any weights
        """
        # task, domain, ts_curr_state, ts_next_state, ts_lbl_vars, ts_boxes, ts_locs  = self.create_symbolic_causal_graph(draw_causal_graph=False)

        # task, domain, ts_curr_state, ts_next_state, ts_gripper_var, ts_on_vars, ts_holding_vars, seg_preds  = self.create_symbolic_causal_graph(draw_causal_graph=False)

        task, domain, ts_sym_vars, seg_preds = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph)

        sym_tr = SymbolicFrankaTransitionSystem(sym_vars_dict=ts_sym_vars,
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