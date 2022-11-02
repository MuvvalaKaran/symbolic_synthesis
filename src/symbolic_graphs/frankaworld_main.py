import re
import sys
import math
import warnings
import copy

from bidict import bidict
from itertools import product
from itertools import combinations
from collections import defaultdict
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

        # maps individual predicates to a unique int
        self.pred_int_map: bidict = bidict({})
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
         A function that create only one set of vars for the objects passed. This function does not create prime varibables. 
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

    
    def _create_all_holding_to_loc_combos(self, predicate_dict: dict)-> List[tuple]:
        """
         A helper function that creates all the valid combinations of holding and to-loc predicates. 

         A valid combination is one where holding's box and location arguements are same as
         to-loc's box and location arguement. 
        """
        _valid_combos = []
        for b in predicate_dict['holding'].keys():
            for l in predicate_dict['holding'][b].keys():
                _valid_combos.extend(list(product(predicate_dict['holding'][b][l], predicate_dict['to_loc'][b][l])))
        
        return _valid_combos


    def _create_all_ready_to_obj_combos(self, predicate_dict: dict) -> List[tuple]:
        """
         A helper function that creates all the valid combinations of ready and to-obj predicates. 

         A valid combination is one where ready location arguement is same as to-obj location arguement. 
        """
        _valid_combos = []
        for key in predicate_dict['ready'].keys():
            if key != 'else':
                _valid_combos.extend(list(product(predicate_dict['ready'][key], predicate_dict['to_obj'][key])))

        return _valid_combos
    

    def _get_all_box_combos(self, boxes_dict: dict, predicate_dict: dict) -> Dict[str, list]:
        """
        The franka world has the world configuration (on b# l#) embedded into it's state defination. 
        Also, we could have all n but 1 boxes grounded with that single box (not grounded) being currently manipulated.
        
        Thus, a valid set of state labels be
            1) all boxes grounded - (on b0 l0)(on b1 l1)...
            2) all but 1 grounded - (on b0 l0)(~(on b1 l1))(on b2 l2)...
        
        Hence, we need to create enough Boolean variables to accomodate all these possible configurations.
        """
        parent_combo_list = {'nb': [],   # preds where all boxes are grounded ad gripper free
                             'b': []     # preds where n-1 boxes are grounded
                             }

        # create all grounded configurations
        all_preds = [val for _, val in boxes_dict.items()]
        all_preds += [predicate_dict['gripper']]
        all_combos = list(product(*all_preds, repeat=1))

        # when all the boxes are grouded then the gripper predicate is set to free
        # parent_combo_list.extend(list(product(all_combos, predicate_dict['gripper'])))
        parent_combo_list['nb'].extend(all_combos)


        # create n-1 combos
        num_of_boxes = len(boxes_dict)

        if num_of_boxes - 1 == 1:
            return parent_combo_list
        
        # create all possible n-1 combinations of all boxes thhat can be grounded
        combos = combinations([*boxes_dict.keys()], num_of_boxes - 1)

        # iterate through every possible combo
        for combo in combos:
            # iterate through the tuple of boxes and create their combos
            box_loc_list = [boxes_dict[box] for box in combo]
            parent_combo_list['b'].extend(list(product(*box_loc_list, repeat=1)))

        return parent_combo_list
    

    def compute_valid_franka_state_tuples(self, robot_preds: Dict[str, list], on_preds: Dict[str, list], verbose: bool = False) -> list:
        """
         A function that take the cartesian prodict of all possbile robot states with all possible world configrations

         robot_preds: all ready, holding predicates along with valid (holding, to-loc) and (ready, to-obj) predicates
         on_preds: all possible grounded (n boxes with their location and gripper free) predicates and n-1 grounded predicates 
        
        The cartesian product gives all possible states of the Franka abstraction.
        """
        # there two typs of prodct, -robot conf where gripper free will have all boxes grounded
        # -robot ocnf where gripper is not free will have n-1 boxes grounded

        _valid_combos_free = list(product(robot_preds['gfree'], on_preds['nb']))
        _valid_combos_occ = list(product(robot_preds['gocc'], on_preds['b']))
        _valid_combos = _valid_combos_free + _valid_combos_occ

        if verbose:
            print(f"********************************* No. Valid States in Frank abstraction: {len(_valid_combos)} *********************************")

        _state_tuples = []
        for _exp_state in _valid_combos:
            _state_tpl = []
            for pred in _exp_state:
                if isinstance(pred, tuple):
                    tmp_tuple = [self.pred_int_map[_s] for _s in pred]
                else:
                    tmp_tuple = [self.pred_int_map[pred]]
                _state_tpl.extend(tmp_tuple) 
        
            _state_tuples.append(tuple(_state_tpl))

        return _state_tuples
    

    def compute_valid_predicates(self, predicates: List[str], boxes: List[str]) -> Tuple[List, List]:
        """
        A helper function that segretaes the predicates as required by the symbolic transition relation. We separate them based on

         1) all gripper predicates - we do not need to create prime version for these
         2) all on predicates - we do need to create prime version for these
         3) all holding predicates 
         4) rest of the predicates - holding, ready, to-obj, and to-loc predicates. We do create prime version for these. 
        """

        predicate_dict = {
            'ready': defaultdict(lambda: []),
            'to_obj': defaultdict(lambda: []),
            'to_loc': defaultdict(lambda: defaultdict(lambda: [])),
            'holding': defaultdict(lambda: defaultdict(lambda: [])),
            'ready_all': [],
            'holding_all': [],
            'to_obj_all': [],
            'to_loc_all': [],
            'on': [],
            'gripper': []
        }

        # dictionary where we segreate on predicates based on boxes - all b0, b1 ,... into seperate list 
        boxes_dict = {box: [] for box in boxes} 

        # define pattaerns to find box ids and locations
        _loc_pattern = "[l|L][\d]+"
        _box_pattern = "[b|B][\d]+"

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
                # ready predicate is not parameterized by box
                if not 'ready' in pred:
                    _box_state: str = re.search(_box_pattern, pred).group()
                    _loc_state: str = re.search(_loc_pattern, pred).group()
                else:
                    # ready predicate can have else as a valid location 
                    if 'else' in pred:
                        _loc_state = 'else'
                    else:
                        _loc_state: str = re.search(_loc_pattern, pred).group()

                if 'holding' in pred:
                    predicate_dict['holding_all'].append(pred)
                    predicate_dict['holding'][_box_state][_loc_state].append(pred)
                elif 'ready' in pred:
                    predicate_dict['ready_all'].append(pred)
                    predicate_dict['ready'][_loc_state].append(pred)
                elif 'to-obj' in pred:
                    predicate_dict['to_obj_all'].append(pred)
                    predicate_dict['to_obj'][_loc_state].append(pred)
                elif 'to-loc' in  pred:
                    predicate_dict['to_loc_all'].append(pred)
                    predicate_dict['to_loc'][_box_state][_loc_state].append(pred)
        
        # create predicate int map
        _ind_pred_list = predicate_dict['ready_all'] + \
             predicate_dict['holding_all'] + predicate_dict['to_obj_all'] + predicate_dict['to_loc_all']
        _pred_map = {pred: num for num, pred in enumerate(_ind_pred_list)}
        _pred_map = bidict(_pred_map)

        # get all valid robot conf predicates
        _valid_robot_preds = {'gfree': [], 
                              'gocc': []}

        # we store valid robot conf into types, one where robot conf exisit when gripper is free and the other robot conf. where gripper is not free
        _valid_robot_preds['gfree'].extend(predicate_dict['ready_all'])
        _valid_robot_preds['gocc'].extend(predicate_dict['holding_all'])

        _valid_robot_preds['gfree'].extend(self._create_all_ready_to_obj_combos(predicate_dict))
        _valid_robot_preds['gocc'].extend(self._create_all_holding_to_loc_combos(predicate_dict))

        # create on predicate map
        len_robot_conf = len(_ind_pred_list)
        _pred_map.update({pred: len_robot_conf + num for num, pred in enumerate(predicate_dict['on'] + predicate_dict['gripper'])})

        # we create all n and n-1 combos
        # n combos when all boxes and gripper is not free are grounded 
        # and n-1 when one of the boxes is being manipulated and gripper is not free
        _valid_box_preds = self._get_all_box_combos(boxes_dict=boxes_dict, predicate_dict=predicate_dict)
        # _valid_box_preds.extend()
        
        # when you have two objects, then individual on predicates are also valid combos 
        if len(_valid_box_preds['b']) == 0:
           _valid_box_preds['b'].extend(predicate_dict['on'])
        
        # predicate_dict['on'].extend(aug_on_state)
        self.pred_int_map = _pred_map
        
        return _valid_robot_preds, _valid_box_preds


    def create_symbolic_causal_graph(self, draw_causal_graph: bool = False, add_flag: bool = False):
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

        task_facts: List[str] = _causal_graph_instance.task.facts
        boxes: List[str] = _causal_graph_instance.task_objects
        valid_locs: List[str] = _causal_graph_instance.task_locations

        # seg_preds = self._segregate_predicates(predicates=task_facts, boxes=boxes)
        # compute all valid preds of the robot conf and box conf.
        robot_preds, on_preds = self.compute_valid_predicates(predicates=task_facts, boxes=boxes)

        ts_state_tuples = self.compute_valid_franka_state_tuples(robot_preds=robot_preds, on_preds=on_preds, verbose=True)

        # compute all the possible states

        # sym_vars = dict(seg_preds)  # shallow copy to avoid changing the org content

        # seg_preds['curr_state'] = seg_preds['others']
        # seg_preds['next_state'] = seg_preds['others']

        # del seg_preds['others']

        ts_lbl_states = self._create_symbolic_lbl_vars(domain_facts=on_preds, state_var_name='b', add_flag=add_flag)
       

        curr_state, next_state = self.create_symbolic_vars(num_of_facts=len(ts_state_tuples),
                                                           add_flag=add_flag)

        # sym_vars['curr_state'] = curr_state
        # sym_vars['next_state'] = next_state
        
        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_state, next_state, ts_state_tuples
        # \ boxes, valid_locs
        

    def build_bdd_abstraction(self, draw_causal_graph: bool = False):
        """
         Main Function to Build Transition System that only represent valid edges without any weights
        """
        # task, domain, ts_sym_vars, seg_preds, boxes, locs = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph)
        task, domain, ts_curr_vars, ts_next_vars, ts_preds  = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph)

        sym_tr = SymbolicFrankaTransitionSystem(curr_states=ts_curr_vars,
                                                next_states=ts_next_vars,
                                                lbl_states=ts_lbl_states,
                                                task=task,
                                                domain=domain,
                                                manager=self.manager,
                                                seg_facts=seg_preds)

        sym_tr.create_transition_system_franka(boxes=boxes,
                                               locs=locs,
                                               add_exist_constr=True,
                                               verbose=False,
                                               plot=self.plot_ts)
        
        sys.exit(-1) 

        return sym_tr, ts_curr_state, ts_next_state, None


    def build_weighted_add_abstraction(self):
        pass