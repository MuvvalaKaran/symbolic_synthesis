import re
import sys
import copy
import math 
import warnings
import graphviz as gv

from typing import Tuple, List, Dict
from collections import defaultdict
from functools import reduce
from cudd import Cudd, BDD, ADD
from itertools import product

from src.explicit_graphs import CausalGraph, FiniteTransitionSystem

from bidict import bidict

from config import *
from utls import *


class SymbolicTransitionSystem(object):
    """
    A class to construct a symbolic transition system for each operator - in our case Actions. 

    curr_state: Symbolic Boolean vairables corresponding to current states
    next_state: Symbolic Boolean vairables corresponding to next states
    lbl_state: Symbolic Boolean vairables corresponding to labels 
    task: Pyperplan object that contains information regarding the task 
    domain : Pyperplan object that contains information regarding the domain
    observations: All the possible observations for a give problem type. For grid world this is same a objects
    manager: CUDD Manager
    """

    def __init__(self, curr_states: list , next_states: list, lbl_states: list, task, domain, manager):
        self.sym_vars_curr = curr_states
        self.sym_vars_next = next_states
        self.sym_vars_lbl = lbl_states
        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.facts:dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        self.manager = manager
        self.actions: dict = domain.actions
        self.tr_action_idx_map: dict = {}
        self.sym_init_states = manager.bddZero()
        self.sym_goal_states = manager.bddZero()
        self.sym_state_labels = manager.bddZero()
        self.sym_tr_actions: list = []
        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl: bidict = {}

        self._create_sym_var_map()
        self._initialize_bdds_for_actions()
        self._initialize_sym_init_goal_states()

    
    def _initialize_bdds_for_actions(self):
        """
        A function to intialize bdds for all the actions
        """
        #  initiate BDDs for all the action 
        action_idx_map = bidict()
        _actions = list(self.actions.keys())
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [self.manager.bddZero() for _ in range(len(self.actions))]
    
    def _initialize_sym_init_goal_states(self):
        """
        Initialize the inital states of the Transition system with their corresponding symbolic init state vairants.
        """
        _init_list = [self.predicate_sym_map_curr.get(s) for s in list(self.init)]
        _goal_list = [self.predicate_sym_map_curr.get(s) for s in list(self.goal)]

        self.sym_init_states = reduce(lambda a, b: a | b, _init_list) 
        self.sym_goal_states = reduce(lambda a, b: a | b, _goal_list)

    
    def build_actions_tr_func(self, curr_edge_action: str, action_list: List):
        """
        A function to build a symbolic transition function corresponding to each action 

        Del effects: If there are multiple del effects then we take the union of them all (del_1 | del_2)
          Then, we take the negation to set them to False 
        
        Add effects: If there are multiple del effects then we take the union of them all (add_1 | add_2)
          The, we take the intersection of Del and Add to ensure that the del effects are not included
        
        pre effects: these are the preconditions that needs to be sarifies in order for the transition to exists.
         We take the union of all the pre conditions and then the intersection of pre with (add and ~del) effects.  
        """

        # since pre, post condition are all forzenset, we iterate over it
        pre_conds = tuple(curr_edge_action.preconditions)
        add_effects = tuple(curr_edge_action.add_effects)
        del_effects = tuple(curr_edge_action.del_effects)

        # get the bool formula for the above predicates
        pre_list = [self.predicate_sym_map_curr.get(pre_cond) for pre_cond in pre_conds]
        add_list = [self.predicate_sym_map_nxt.get(add_effect) for add_effect in add_effects]
        del_list = [self.predicate_sym_map_nxt.get(del_effect) for del_effect in del_effects]

        # if multiple then the conditions is the union
        if len(pre_list) != 0:
            pre_sym = reduce(lambda a, b: a | b, pre_list) 
        else:
            pre_sym = self.manager.bddOne()
        if len(add_list) != 0:
            add_sym = reduce(lambda a, b: a | b, add_list)
        else:
            add_sym = self.manager.bddOne()
        if len(del_list) != 0:
            del_sym = reduce(lambda a, b: a | b, del_list)
        else:
            del_sym = self.manager.bddZero()
        
        _curr_action_name = curr_edge_action.name

        # instead looking for the action, extract it (action name)
        _action = _curr_action_name.split()[0]
        _action = _action[1:]   # remove the intial '(' braket

        # assert that its a valid name
        assert _action in action_list, "FIX THIS: Failed extracting a valid action."

        # pre_sym - precondition will be false when starting from the initial states; means you can take the action under all conditions
        if pre_sym.isZero():
            pre_sym = pre_sym.negate()
        
        if add_sym.isZero():
            add_sym = add_sym.negate()

        _idx = self.tr_action_idx_map.get(_action)
        self.sym_tr_actions[_idx] |= pre_sym & add_sym & ~del_sym
        
        return _action
    

    def _create_sym_state_label_map(self, domain_lbls):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it.
         
         This method is called whten Gridworld state labels are created
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_lbl)))

        # We will add dumy brackets around the label e.g. l4 ---> (l4) becuase promela parse names the edges in that fashion
        _node_int_map_lbl = bidict({state: boolean_str[index] for index, state in enumerate(domain_lbls)})

        assert len(boolean_str) >= len(_node_int_map_lbl), "FIX THIS: Looks like there are more lbls that boolean variables!"

        # loop over all the boolean string and convert them to their respective bdd vars
        for _key, _value in _node_int_map_lbl.items():
            _lbl_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _lbl_val_list.append(self.sym_vars_lbl[_idx])
                else:
                    _lbl_val_list.append(~self.sym_vars_lbl[_idx])
            
            _bool_func_curr = reduce(lambda a, b: a & b, _lbl_val_list)

            # update bidict accordingly
            _node_int_map_lbl[_key] = _bool_func_curr
        
        self.predicate_sym_map_lbl = _node_int_map_lbl


    def _create_sym_var_map(self):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """

        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_curr)))

        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(self.facts)})
        _node_int_map_next = copy.deepcopy(_node_int_map_curr)

        assert len(boolean_str) >= len(_node_int_map_next), "FIX THIS: Looks like there are more Facts that boolean variables!"

        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map_curr.items():
            _curr_val_list = []
            _next_val_list = []
            # _bool_fun = self.manager.bddOne()
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    # _bool_func = _bool_fun & self.sym_vars_curr[_idx]
                    _curr_val_list.append(self.sym_vars_curr[_idx])
                    _next_val_list.append(self.sym_vars_next[_idx])
                else:
                    _curr_val_list.append(~self.sym_vars_curr[_idx])
                    _next_val_list.append(~self.sym_vars_next[_idx])
            
            _bool_func_curr = reduce(lambda a, b: a & b, _curr_val_list)
            _bool_func_nxt = reduce(lambda a, b: a & b, _next_val_list)

            # update bidict accordingly
            _node_int_map_curr[_key] = _bool_func_curr
            _node_int_map_next[_key] = _bool_func_nxt    
        
        self.predicate_sym_map_curr = _node_int_map_curr
        self.predicate_sym_map_nxt = _node_int_map_next
    


    def create_transition_system(self, verbose: bool = False, plot: bool = False):
        """
        A function to create the TR function for each Action we hav defined in the domain
        """
        if verbose:
            print(f"Creating TR for Actions {self.domain.actions}")
        
        scheck_count_transit = 0
        scheck_count_transfer = 0
        scheck_count_grasp = 0
        scheck_count_release = 0
        scheck_count_hmove = 0

        action_list = list(self.actions.keys())
        
        for _action in self.task.operators:
            self.build_actions_tr_func(curr_edge_action=_action, action_list=action_list)

        if verbose:
            for _action, _idx in self.tr_action_idx_map.items():
                print(f"Charateristic Function for action {_action} \n")
                print(self.sym_tr_actions[_idx], " \n")
                if plot:
                    file_path = PROJECT_ROOT + f'/plots/{_action}_trans_func.dot'
                    file_name = PROJECT_ROOT + f'/plots/{_action}_trans_func.pdf'
                    self.manager.dumpDot([self.sym_tr_actions[_idx]], file_path=file_path)
                    gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)

        
            print("# of total symbolic transfer edge counts", scheck_count_transit)
            print("# of total symbolic transit edge counts", scheck_count_transfer)
            print("# of total symbolic grasp edge counts", scheck_count_grasp)
            print("# of total symbolic release edge counts", scheck_count_release)
            print("# of total symbolic human-move edge counts", scheck_count_hmove)
            print("All done!")
            

    
    def create_state_obs_bdd(self, domain_lbls,  verbose: bool = False, plot: bool = False):
        """
        A function to create the Charactersitic function for each state observation in the domain. 
        """
        self._create_sym_state_label_map(domain_lbls=domain_lbls)
        # A state looks like - `at skbn l#` and we are trying extract l# and add that as bdd
        for state in self.facts:
            split_str = re.split(" ", state)
            _loc = split_str[-1][:-1]   # remove the trailing bracket

            # look up the correspondig boolean formula associated with 
            _state_bdd = self.predicate_sym_map_curr.get(state)
            _obs_bdd = self.predicate_sym_map_lbl.get(_loc)

            assert _obs_bdd or _state_bdd is not None, "Looks like we extracted na invalid observation. FIX THIS!!!"

            self.sym_state_labels |= _state_bdd & _obs_bdd
        
        if verbose:
            print("*************Printing State Observation BDD*************")
            print(self.sym_state_labels)
            if plot: 
                file_path = PROJECT_ROOT + f'/plots/S2Obs_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/S2Obs_trans_func.pdf'
                self.manager.dumpDot([self.sym_state_labels], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)



class SymbolicWeightedTransitionSystem(object):
    """
    A class to construct a symbolic Weighted transition system for each operator - in our case Actions. 

    curr_state: Symbolic Boolean vairables corresponding to current states
    next_state: Symbolic Boolean vairables corresponding to next states
    lbl_state: Symbolic Boolean vairables corresponding to labels 
    task: Pyperplan object that contains information regarding the task 
    domain : Pyperplan object that contains information regarding the domain
    observations: All the possible observations for a give problem type. For grid world this is same a objects
    manager: CUDD Manager
    weight_dict: A mapping from pddl action to their corresponding weights in the dictionary. 

    We had to create a dedicated class for this as allthe variables are ad variables in the this class.
    This includes, states, transition function, and labels 
    """

    def __init__(self, curr_states: List[ADD] , next_states: List[ADD], lbl_states: List[ADD], task, domain, weight_dict, manager: Cudd):
        self.sym_add_vars_curr: List[ADD] = curr_states
        self.sym_add_vars_next: List[ADD] = next_states
        self.sym_add_vars_lbl: List[ADD] = lbl_states
        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.facts:dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        # self.domain_lbls = observations
        self.manager: Cudd = manager
        self.weight_dict: Dict[str, List[ADD]] = weight_dict

        self.tr_action_idx_map: dict = {}
        self.sym_add_init_states = manager.addZero()
        self.sym_add_goal_states = manager.addZero()
        self.sym_add_state_labels = manager.addZero()
        self.sym_tr_actions: list = []
        self.predicate_add_sym_map_curr: bidict = {}
        self.predicate_add_sym_map_nxt: bidict = {}
        self.predicate_add_sym_map_lbl: bidict = {}

        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl: bidict = {}

        self._create_sym_var_map()
        self._initialize_adds_for_actions()
        self._initialize_sym_init_goal_states()
    

    def _initialize_adds_for_actions(self):
        """
        A function to intialize bdds for all the actions
        """
        #  initiate BDDs for all the action 
        action_idx_map = bidict()
        _actions = list(self.weight_dict.keys())
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [self.manager.addZero() for _ in range(len(self.weight_dict))]
    

    def _initialize_sym_init_goal_states(self):
        """
        Initialize the inital states of the Transition system with their corresponding symbolic init state vairants.
        """
        _init_list = [self.predicate_add_sym_map_curr.get(s) for s in list(self.init)]
        _goal_list = [self.predicate_add_sym_map_curr.get(s) for s in list(self.goal)]

        self.sym_add_init_states = reduce(lambda a, b: a | b, _init_list) 
        self.sym_add_goal_states = reduce(lambda a, b: a | b, _goal_list)
    

    def _create_sym_var_map(self):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """

        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_add_vars_curr)))

        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(self.facts)})
        _node_int_map_next = copy.deepcopy(_node_int_map_curr)

        _node_bdd_int_map_curr = copy.deepcopy(_node_int_map_curr)
        _node_bdd_int_map_next = copy.deepcopy(_node_int_map_curr)

        assert len(boolean_str) >= len(_node_int_map_next), "FIX THIS: Looks like there are more Facts that boolean variables!"

        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map_curr.items():
            _curr_val_list = []
            _next_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _curr_val_list.append(self.sym_add_vars_curr[_idx])
                    _next_val_list.append(self.sym_add_vars_next[_idx])
                else:
                    _curr_val_list.append(~self.sym_add_vars_curr[_idx])
                    _next_val_list.append(~self.sym_add_vars_next[_idx])
            
            # we will create two dictionary, one mapping add to curr and nxt state
            # the other dictionary will make the corresponding bdd to curr and nxt state
            _bool_add_func_curr = reduce(lambda a, b: a & b, _curr_val_list)
            _bool_add_func_nxt = reduce(lambda a, b: a & b, _next_val_list)

            _bool_bdd_func_curr = _bool_add_func_curr.bddPattern()
            _bool_bdd_func_nxt = _bool_add_func_nxt.bddPattern()

            # update bidict accordingly
            _node_int_map_curr[_key] = _bool_add_func_curr
            _node_int_map_next[_key] = _bool_add_func_nxt

            _node_bdd_int_map_curr[_key] = _bool_bdd_func_curr
            _node_bdd_int_map_next[_key] = _bool_bdd_func_nxt 
        
        self.predicate_add_sym_map_curr = _node_int_map_curr
        self.predicate_add_sym_map_nxt = _node_int_map_next

        self.predicate_sym_map_curr = _node_bdd_int_map_curr
        self.predicate_sym_map_nxt = _node_bdd_int_map_next  


    def _create_sym_state_label_map(self, domain_lbls):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_add_vars_lbl)))

        _node_int_map_lbl = bidict({state: boolean_str[index] for index, state in enumerate(domain_lbls)})
        _node_bdd_int_map_lbl = copy.deepcopy(_node_int_map_lbl)

        assert len(boolean_str) >= len(_node_int_map_lbl), "FIX THIS: Looks like there are more lbls that boolean variables!"

        # loop over all the boolean string and convert them to their respective bdd vars
        for _key, _value in _node_int_map_lbl.items():
            _lbl_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _lbl_val_list.append(self.sym_add_vars_lbl[_idx])
                else:
                    _lbl_val_list.append(~self.sym_add_vars_lbl[_idx])
            
            _bool_add_func_curr = reduce(lambda a, b: a & b, _lbl_val_list)
            _bool_bdd_func_curr = _bool_add_func_curr.bddPattern()

            # update bidict accordingly
            _node_int_map_lbl[_key] = _bool_add_func_curr
            _node_bdd_int_map_lbl[_key] = _bool_bdd_func_curr
        
        self.predicate_add_sym_map_lbl = _node_int_map_lbl
        self.predicate_sym_map_lbl = _node_bdd_int_map_lbl
    

    def create_weighted_transition_system(self, verbose: bool = False, plot: bool = False):
        """
        A function to create the TR function for each Action we have defined in the domain along with its weight
        """
        if not all(isinstance(tr_const, ADD) for tr_const in self.weight_dict.values()):
            print("Please Make sure your edge weights are of type ADD. FIX THIS!!!")
            sys.exit()

        for _action in self.task.operators:
            self.build_actions_tr_func_weighted(curr_edge_action=_action)

        if verbose:
            for _action, _idx in self.tr_action_idx_map.items():
                print(f"Charateristic Function for action {_action} \n")
                print(self.sym_tr_actions[_idx], " \n")
                if plot:
                    file_path = PROJECT_ROOT + f'/plots/{_action}_ADD_trans_func.dot'
                    file_name = PROJECT_ROOT + f'/plots/{_action}_ADD__trans_func.pdf'
                    self.manager.dumpDot([self.sym_tr_actions[_idx]], file_path=file_path)
                    gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)
    

    def build_actions_tr_func_weighted(self, curr_edge_action: str):
        """
        A function to build a symbolic transition function corresponding to each action along with its weight
        """

        _actions = list(self.weight_dict.keys())
        # since pre, post condition are all forzenset, we iterate over it
        pre_conds = tuple(curr_edge_action.preconditions)
        add_effects = tuple(curr_edge_action.add_effects)
        del_effects = tuple(curr_edge_action.del_effects)

        # get the bool formula for the above predicates in terms of add variables
        # Note add in the predictice_add... is not the same as addition but it is Algebraic Decision Diagram
        pre_list = [self.predicate_add_sym_map_curr.get(pre_cond) for pre_cond in pre_conds]
        add_list = [self.predicate_add_sym_map_nxt.get(add_effect) for add_effect in add_effects]
        del_list = [self.predicate_add_sym_map_nxt.get(del_effect) for del_effect in del_effects]

        if len(pre_list) != 0:
            pre_sym = reduce(lambda a, b: a & b, pre_list)
        else:
            pre_sym = self.manager.bddOne()
        if len(add_list) != 0:
            add_sym = reduce(lambda a, b: a & b, add_list)
        else:
            add_sym = self.manager.bddOne()
        if len(del_list) != 0:
            del_sym = reduce(lambda a, b: a & b, del_list)
        else:
            del_sym = self.manager.bddZero()
        
        _curr_action_name = curr_edge_action.name

        # instead looking for the action, extract it (action name)
        _action = _curr_action_name.split()[0]
        _action = _action[1:]   # remove the intial '(' braket

        # assert that its a valid name
        assert _action in _actions, "FIX THIS: Failed extracting a valid action."

        # pre_sym - precondition will be false when starting from the initial states; mean you can take the action under all conditions
        if pre_sym.isZero():
            pre_sym = pre_sym.negate()
        
        if add_sym.isZero():
            add_sym = add_sym.negate()

        _idx = self.tr_action_idx_map.get(_action)  

        self.sym_tr_actions[_idx] |= pre_sym & add_sym & ~del_sym & self.weight_dict[_action]
        
        return _action
    

    def create_state_obs_add(self, domain_lbls, verbose: bool = False, plot: bool = False):
        """
        A function to create the Charactersitic function for each state observation in the domain. 
        """
        self._create_sym_state_label_map(domain_lbls=domain_lbls)
        # this onlt works for grid world for now. A state looks like - `at skbn l#` and we are trying extract l# and add that as bdd
        for state in self.facts:
            split_str = re.split(" ", state)
            _loc = split_str[-1][:-1]   # remove the trailing bracket

            # look up the correspondig boolean formula associated with 
            _state_add = self.predicate_add_sym_map_curr.get(state)
            _obs_add = self.predicate_add_sym_map_lbl.get(_loc)

            assert _obs_add or _state_add is not None, "Looks like we extracted an invalid observation. FIX THIS!!!"

            self.sym_add_state_labels |= _state_add & _obs_add
        
        if verbose:
            print("*************Printing State Observation BDD*************")
            print(self.sym_add_state_labels)
            if plot: 
                file_path = PROJECT_ROOT + f'/plots/S2Obs_ADD_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/S2Obs_ADD_trans_func.pdf'
                self.manager.dumpDot([self.sym_add_state_labels], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)


class SymbolicFrankaTransitionSystem():
    """
     A class to construct the symblic transition system for the Robotic manipulator example.
    """

    def __init__(self, sym_vars_dict: dict, task, domain, manager: Cudd, seg_facts: dict):
        # self.sym_vars_curr = curr_states
        # self.sym_vars_next = next_states
        # self.sym_vars_lbl = lbl_states
        # self.sym_gripper_var = gripper_var
        # self.sym_on_vars = on_vars
        # self.sym_holding_vars = holding_vars
        self.sym_vars_dict: dict = sym_vars_dict
        self.seg_facts_dict: dict = seg_facts 

        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.facts: dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        self.manager = manager
        self.actions: dict = domain.actions
        self.tr_action_idx_map: dict = {}
        self.sym_init_states = manager.bddZero()
        self.sym_goal_states = manager.bddZero()
        self.sym_state_labels = manager.bddZero()
        self.sym_tr_actions: list = []
        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl: bidict = {}
        
        # parent diction with all current and state lbl vars
        self.monolihtic_sym_map_curr: bidict = {}

        # parent dictions with all the nxt and state lbl vars
        self.monolihtic_sym_map_nxt: bidict = {}

        self._create_sym_var_map()
        self._initialize_bdds_for_actions()
        self._initialize_sym_init_goal_states()

    def _initialize_bdds_for_actions(self):
        """
        A function to intialize bdds for all the actions
        """
        #  initiate BDDs for all the action 
        action_idx_map = bidict()
        _actions = list(self.actions.keys())
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [self.manager.bddZero() for _ in range(len(self.actions))]
    

    def _initialize_sym_init_goal_states(self):
        """
        Initialize the inital states of the Transition system with their corresponding symbolic init state vairants.
        """
        on_state_conf = []   # where the boxes are placed
        _init_list = []
        for istate in list(self.init) :
            if 'ready' in istate:
                _init_list.append(self.predicate_sym_map_curr[istate])
            elif 'on' in istate or 'gripper' in istate:
                on_state_conf.append(self.predicate_sym_map_lbl[istate])
            else:
                warnings.warn("Error while creating the initial state. Encountered unexpect predicae. FIX THIS!!!")
                sys.exit(-1)
        
        self.sym_init_states = reduce(lambda a, b: a & b, _init_list) & reduce(lambda a, b: a | b, on_state_conf)
        _goal_list = [self.predicate_sym_map_lbl[s] for s in self.goal]
        
        # we take the union as the goal list is completely defined by all on predicates
        self.sym_goal_states = reduce(lambda a, b: a | b, _goal_list)   
    

    def _create_sym_var_map(self):
        """
         A function that initialize the dictionary that map every ground facts to its corresponding boolean formula.  
        """

        for pred_type, pred_list in self.seg_facts_dict.items():
            pred_dict = self._create_sym_var_map_per_fact(facts=pred_list, pred_type=pred_type)
            if pred_type == 'curr_state':
                self.predicate_sym_map_curr.update(pred_dict)
                self.monolihtic_sym_map_curr.update(self.predicate_sym_map_curr)
            elif pred_type == 'next_state':
                self.predicate_sym_map_nxt.update(pred_dict)
                self.monolihtic_sym_map_nxt.update(self.predicate_sym_map_nxt)
            else:
                self.predicate_sym_map_lbl.update(pred_dict)
                self.monolihtic_sym_map_curr.update(self.predicate_sym_map_lbl)
                self.monolihtic_sym_map_nxt.update(self.predicate_sym_map_lbl)
        
        self.predicate_sym_map_curr = bidict(self.predicate_sym_map_curr)
        self.predicate_sym_map_lbl = bidict(self.predicate_sym_map_lbl)
        self.predicate_sym_map_nxt = bidict(self.predicate_sym_map_nxt)
        

    def _create_sym_var_map_per_fact(self, facts: List[str], pred_type: str):
        """
         Loop through all the facts and assign a boolean funtion to it
        """

        # create all combinations of 1-true and 0-false; choose the appropriate length of the boolean formula based on the caterogy of predicate
        sym_vars = self.sym_vars_dict[pred_type]
        boolean_str = list(product([1, 0], repeat=len(sym_vars)))
        
        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(facts)})

        assert len(boolean_str) >= len(_node_int_map_curr), "FIX THIS: Looks like there are more Facts that boolean variables!"

        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map_curr.items():
            _curr_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _curr_val_list.append(sym_vars[_idx])
                else:
                    _curr_val_list.append(~sym_vars[_idx])
                  
            _bool_func_curr = reduce(lambda a, b: a & b, _curr_val_list)

            # update bidict accordingly
            _node_int_map_curr[_key] = _bool_func_curr

        
        return _node_int_map_curr

    
    def print_state_lbl_dd(self, dd_func: BDD, ts_x_cube: BDD, lbl_cube: BDD) -> Tuple[List[str], List[str]]:
        """
        This function wraps around __convert_state_lbl_cube_to_func(), compute all the possible cubes,
         looks up their correpsonding state name and prints it.  
        """
        only_states: BDD = dd_func.existAbstract(lbl_cube)
        # only_lbls: BDD = dd_func.existAbstract(ts_x_cube)

        # testing 
        # prod_cube = self._convert_state_lbl_cube_to_func(dd_func=dd_func, prod_curr_list=self.sym_vars_dict['curr_state'] + self.sym_vars_dict['on'])

        state_cube_string: List[BDD] = self._convert_state_lbl_cube_to_func(dd_func=only_states, prod_curr_list=self.sym_vars_dict['curr_state'])
        
        # for each of those states, extract their corresponding labels
        s_lbl_list = []
        for scube in state_cube_string:
            s_lbl_dd = dd_func.restrict(scube)
            if s_lbl_dd.isOne():
                # print("WARNING: Got a state where none of the boxes are grounded. This should only happen when you have one object!")
                continue
            s_lbl_cube: List[BDD] = self._convert_state_lbl_cube_to_func(dd_func=s_lbl_dd, prod_curr_list=self.sym_vars_dict['on'])
            s_lbl_list = [self.predicate_sym_map_lbl.inv.get(sym_s, None) for sym_s in s_lbl_cube]
        # lbl_cube_string: List[BDD] = self._convert_state_lbl_cube_to_func(dd_func=only_lbls, prod_curr_list=self.sym_vars_dict['on'])
        states = [self.predicate_sym_map_curr.inv[sym_s] for sym_s in state_cube_string]
        # lbls = [self.predicate_sym_map_lbl.inv[sym_s] for sym_s in lbl_cube_string]
        lbls = s_lbl_list

        return states, lbls


    def _convert_state_lbl_cube_to_func(self, dd_func: BDD, prod_curr_list = None) ->  List[BDD]:
        """
         A helper function to extract a cubes from the DD and print them in human-readable form. 
        """
        ddVars = []
        for cube in dd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                # skip the primed variables
                if var == 2 and self.manager.bddVar(_idx) not in prod_curr_list:   # not x list is better than y _list because we also have dfa vairables 
                    continue   # skipping over prime states 
                else:
                    if var == 2:
                        _amb_var.append([self.manager.bddVar(_idx), ~self.manager.bddVar(_idx)])   # count how many vars are missing to fully define the bdd
                    elif var == 0:
                        var_list.append(~self.manager.bddVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.bddVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                        sys.exit(-1)
                
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    ddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                ddVars.append(reduce(lambda a, b: a & b, var_list))
        
        return ddVars


    
    def _get_sym_conds(self, conditions: List[str], nxt_state_flag: bool = False) -> BDD:
        """
        A function that constructs a BDD associated with the list of conditions passed (could b pre, add, or delete)
         and return conjoined BDD 
        
        For the delete and add symbols, we return the next state variables by setting the nxt_state_flag to True. 
        """
        _on_sym = self.manager.bddZero()
        _state_sym = self.manager.bddZero()
        for state in conditions:
            if 'gripper' in state or 'on' in state:
                _on_sym |= self.predicate_sym_map_lbl[state]
            else:
                if nxt_state_flag:
                    _state_sym |= self.predicate_sym_map_nxt[state]
                else:
                    _state_sym |= self.predicate_sym_map_curr[state]
        
        if _state_sym.isZero():
            # this will never happen for our current Franka PDDL file
            _state_sym |= self.manager.bddOne()
        if _on_sym.isZero():
            _on_sym |= self.manager.bddOne()

        return _state_sym & _on_sym

    
    def __pre_in_state(self, conditions: List, curr_states: BDD):
        """
        A method that check if all the pre conditions have been met or not. 
        """
        # check if all the preconditions are met or not.
        _intersect = self.manager.bddOne()
        for pre_s in conditions:
            if 'on' in pre_s or 'gripper' in pre_s:
                _intersect = curr_states & self.predicate_sym_map_lbl[pre_s]
            else:
                _intersect = curr_states & self.predicate_sym_map_curr[pre_s]
            
            if _intersect.isZero():
                return False
        return True
    

    def _check_exist_constraint(self, boxes: List[str], curr_state_lbl: BDD, action_name: str) -> bool:
        """
        A helper function that take as input the state label (on b0 l0)(on b1 l1) and the action name,
         extracts the destination location from action name and its corresponding symbolic formula.
         We then take the intersection of the  sym_state_lbl & not(exist_constraint).
         
        Return True if intersection is non-empty else False
        """
        finite_ts = FiniteTransitionSystem(None)
        exist_constr = self.manager.bddZero()
        if 'transfer' in action_name: 
            box_id, locs = finite_ts._get_multiple_box_location(multiple_box_location_str=action_name)
            dloc = locs[1]
        elif 'release' in action_name:
            box_id, locs = finite_ts._get_box_location(box_location_state_str=action_name)
            dloc = locs
        
        # if box1 is being manipulated, get the list of rest of boxes (that have to be grounded) at this instance
        tmp_copy = copy.deepcopy(boxes)
        tmp_copy.remove(f'b{box_id}')

        # create predicates that say on b0 l1 (l1 is the destination in the action)
        on_preds = [self.predicate_sym_map_lbl[f'(on {bid} {dloc})'] for bid in tmp_copy]
        sym_constr = reduce(lambda a, b: a | b, on_preds)

        _is_valid = curr_state_lbl & ~sym_constr

        if _is_valid.isZero():
            return False
        
        return True


    def _compute_valid_states(self, preconditions: List[str], curr_states: BDD, lbl_cube: BDD, locations: List[str]) -> BDD:
        """
        A function that compute the set of states that strictly satify the preconditions. 

        Since out state lbsl and state vars are made dijoint vars, and have mutliple lbls define a state, we take the union
        i.e., (b0 | b1 | b2) & (x0 | x2) where, say, b0, b1, b2 correspond to boxes 0,1, and 2 placed on loc 0, 1, 2.
         Similarly, x0 and x1 respresent robot "holding b# l#" and "to-loc b# l#". 

        From this we valid actions are release and transfer accroding to our PDDL file. 

        Given a set of states S, to compute the set of states S' that satisfy preconditions  of release action, we take the
         intersection of each precondition and take the intersection of the intersections to compute S'. 
        """
        intr_state: List[BDD] = []  # to keep track the world conf that satifies each precond. individually
        lbl_states: List[str] = []  # bookkeeping the state labels which are used to check of valid world conf. for a given action
       
        for state in preconditions:
            # state that satisfies a state conf pre configuration
            if 'gripper' in state or 'on' in state:
                intr_state.append((self.predicate_sym_map_lbl[state] & curr_states).existAbstract(lbl_cube))
                lbl_states.append(state)
            # state that satistfies the pre conditions of the conf. of the  robot 
            else:
                intr_state.append((self.predicate_sym_map_curr[state] & curr_states).existAbstract(lbl_cube))
    
        _valid_pre_state = reduce(lambda x, y: x & y, intr_state)
        
        if _valid_pre_state.isZero():
            # when only state conf are in the preconditions, intersection will give us those preconditions individually,
            # thus we take the union
            _single_state_sym = self.manager.bddZero() 
            _multi_state_sym = self.manager.bddOne()
            for _s in intr_state:
                # hacky way to check is it one single state or multiple
                if self.predicate_sym_map_curr.inv.get(_s, None):
                    _single_state_sym |= _s
                else:
                    _multi_state_sym = _multi_state_sym & _s
                    

            _valid_pre_state = _single_state_sym & _multi_state_sym

        # get all the assocated world conf
        _conf: BDD = curr_states.restrict(_valid_pre_state)
        
        # remove the invalid world conf.
        _pre_lbl = []
        _loc_pattern = "[l|L][\d]+"
        _box_pattern = "[b|B][\d]+"
        for _lbl in lbl_states:
            # extract gripper status and current box id and loc
            if 'free' in _lbl:
                _pre_lbl.append('free')
            else:
                _loc_state: str = re.search(_loc_pattern, _lbl).group()
                _box_state: str = re.search(_box_pattern, _lbl).group()
                _pre_lbl.append((_box_state, _loc_state))
        
        # if state lbls (on or gripper predicates) are part of precondition
        if len(_pre_lbl) > 0:
            # get the neg of current gripper status and for invalid box conf. 
            # say precond is (on b0 l1); then invalid conf. is (on b0 l2),(on b0 l0)... etc
            _del_lbl: List[BDD] = []
            _del_loc = copy.deepcopy(locations)
            for _lbl in _pre_lbl:
                if isinstance(_lbl, tuple):
                    _del_loc.remove(_lbl[1])
            
            # create (on b0 l2)(on b0 l1) ...etc sym variants
            _del_on_preds = [self.predicate_sym_map_lbl[f'(on {_box_state} {loc})'] for loc in _del_loc]
            _del_lbl.extend(_del_on_preds)

            _del_sym_constr: BDD = reduce(lambda a, b: a | b, _del_lbl)

            _valid_conf = _conf & ~(_del_sym_constr)

        else:
            _valid_conf = _conf

        _valid_composed_state = _valid_pre_state & _valid_conf
        
        assert not _valid_composed_state.isZero(), \
                "Error computing the set of valid pre states from which any kind of tranistion during Franka abstraction construction. FIX THIS!!!"
        
        return _valid_composed_state
    

    def _update_open_list(self, closed: Dict, open_list: Dict, layer: int, conf: BDD, lbl_cube: BDD) -> Dict:
        """
         A helper function to update the closed list. For actions Grasp and Release, we have a union of state conf. are preconditions. 

         Thus, we check if both the both the state preconditions have met or not. If yes, then return empty else return the entire union.
        """
        # if action.name in ['grasp', 'release']:
        # check if all the state conf exists in the closed list or not
        only_states: BDD = open_list[layer][conf].existAbstract(lbl_cube)
        state_cube_string: List[BDD] = self._convert_state_lbl_cube_to_func(dd_func=only_states, prod_curr_list=self.sym_vars_dict['curr_state'])
        if len(state_cube_string) == 1:
            open_list[layer][conf] = open_list[layer][conf] & ~closed.get(conf, self.manager.bddZero())
            return 
        
        # remove_curr_state: bool = False
        for scube in state_cube_string:
            if not (scube & closed.get(conf, self.manager.bddZero())).isZero():
                # nothing to remove from the open_list
                remove_curr_state = True
            else:
                remove_curr_state = False
                break

        if remove_curr_state:
            open_list[layer][conf] = open_list[layer][conf] & ~closed.get(conf, self.manager.bddZero())
        
        return 

    def create_transition_system_franka(self,
                                        boxes: List[str],
                                        locs: List[str],
                                        add_exist_constr: bool = True,
                                        verbose:bool = False,
                                        plot: bool = False):
        """
         The construction of TR for the franka world is a bit different than the gridworld. In gridworld the
          complete information of the world, i.e., the agent current location is embedded and thus
          create_trasntion_system()'s implementation sufficient when we label the states.

         Unlike gridworld, in franka world, we have to start from the iniital states with the initial conditions
          explicitly mentioned in problem file. From here, we create the state (pre) and its corresponding labels,
          we unroll the graph by taking all valid actions, compute the image, and their corresponding labels and
          keep iterating until we reach a fix point. 
        
        @param add_exist_constr: Adds the existential constraint that
            1) no other box should exist at the destination while performing Transfer action - transfer b# l# l#
            2) no other box should exist at the drop location while performing Relase action - release b# l# 
        """
        if verbose:
            print(f"Creating TR for Actions {self.domain.actions}")
        
        action_list = list(self.actions.keys())

        open_list = {}
        # closed = self.manager.bddZero()
        closed = defaultdict(lambda: self.manager.bddZero())

        init_state = self.sym_init_states
 
        ts_x_cube = reduce(lambda x, y: x & y, self.sym_vars_dict['curr_state'])
        ts_y_cube = reduce(lambda x, y: x & y, self.sym_vars_dict['next_state'])
        lbl_cube = reduce(lambda x, y: x & y, self.sym_vars_dict['on'])

        # extract the labels out 
        state_lbls = init_state.existAbstract(ts_x_cube)

        layer = 0
        open_list[layer] = {state_lbls: init_state}
        # open_list[layer] = init_state

        # no need to check if other boxes are placed at the destination loc during transfer and release as there is only one object
        if len(boxes) == 1:
            add_exist_constr = False

        # start from the initial conditions
        # while not open_list[layer].isZero():
        while True:
            if not layer in open_list:
                print("******************************* Reached a Fixed Point *******************************")
                break
            
            if verbose:
                print(f"******************************* Layer: {layer}*******************************")
            for conf in open_list[layer].keys():
                # remove all states that have been explored
                # open_list[layer] = open_list[layer] & ~closed
                # open_list[layer][conf] = open_list[layer][conf] & ~closed.get(conf, self.manager.bddZero())
                self._update_open_list(closed=closed,
                                       open_list=open_list,
                                       layer=layer,
                                       conf=conf,
                                       lbl_cube=lbl_cube)

                # If unexpanded states exist ... 
                # for conf, sym_state in open_list[layer]:
                if not open_list[layer][conf].isZero():
                    # Add states to be expanded next to already expanded states
                    closed[conf] |= open_list[layer][conf]

                    # compute the image of the TS states 
                    for action in self.task.operators:
                        # set action feasbility flag to True - used during transfer and release action to check the des loc is empty
                        action_feas: bool = True
                        pre_sym = self._get_sym_conds(list(action.preconditions))
                    
                        # check if there is an intersection 
                        _intersect: bool = self.__pre_in_state(list(action.preconditions), curr_states=open_list[layer][conf])
                        
                        if _intersect:
                            # compute the successor state and their label
                            _valid_pre = self._compute_valid_states(preconditions=list(action.preconditions),
                                                                    curr_states=open_list[layer][conf],
                                                                    locations=locs,
                                                                    lbl_cube=lbl_cube)
                            
                            pre_sym_state = _valid_pre.existAbstract(lbl_cube).swapVariables(self.sym_vars_dict['curr_state'], self.sym_vars_dict['next_state'])

                            # extract the labels out 
                            state_lbls = _valid_pre.existAbstract(ts_x_cube)
                            if state_lbls.isOne():
                                state_lbls = open_list[layer][conf].existAbstract(ts_x_cube)
                            
                            # add existential constraints to transfer and relase action
                            if add_exist_constr and (('transfer' in action.name) or ('release' in action.name)):
                                action_feas = self._check_exist_constraint(boxes=boxes,
                                                                            curr_state_lbl=state_lbls,
                                                                            action_name=action.name)
                            
                            if not action_feas:
                                continue

                            add_sym = self._get_sym_conds(list(action.add_effects), nxt_state_flag=True)
                            del_sym = self._get_sym_conds(list(action.del_effects), nxt_state_flag=True)

                            del_nxt_state_lbls = del_sym.existAbstract(ts_y_cube) #& add_sym.existAbstract(ts_y_cube)
                            add_nxt_state_lbls = add_sym.existAbstract(ts_y_cube)

                            if del_sym.isOne():
                                del_sym = del_sym.negate()

                            # check if there is anything to remove from pre -maybe all or none
                            _del_sym_state_only = del_sym.existAbstract(lbl_cube)
                            _del_pre_intr = _del_sym_state_only & pre_sym_state
                            _pre_sym_state = pre_sym_state & ~(_del_pre_intr)

                            if del_nxt_state_lbls.isOne() and add_nxt_state_lbls.isOne():
                                # nothing to add or delete in the world box conf.
                                next_state_lbls = state_lbls
                                nxt_state = ~del_sym & (add_sym | _pre_sym_state) & next_state_lbls
                            
                            elif add_nxt_state_lbls.isOne():
                                # nothing to add
                                next_state_lbls = state_lbls & ~del_nxt_state_lbls
                                if next_state_lbls.isZero():
                                    print("******************************* None of the objects are grounded!******************************* ")
                                    nxt_state = ~del_sym & (add_sym | _pre_sym_state)
                                else:
                                    nxt_state = ~del_sym & (add_sym | _pre_sym_state) & next_state_lbls
                            # nothing to delete only adding state lables
                            elif del_nxt_state_lbls.isOne():
                                # if the previous state had on boxes grounded then take the intersection
                                if state_lbls.isOne():
                                    next_state_lbls = state_lbls & add_nxt_state_lbls
                                else:
                                    next_state_lbls = state_lbls | add_nxt_state_lbls

                                # extract labels from add as they already exist in next_state_lbls
                                add_sym_state_only = add_sym.existAbstract(lbl_cube)
                                nxt_state = ~del_sym & (add_sym_state_only | _pre_sym_state) & next_state_lbls

                            else:
                                warnings.warn("Error computing the world configurtion during abstraction construction. FIX THIS!!!")
                                sys.exit(-1)
                            
                            if verbose:
                                cstate, clbl = self.print_state_lbl_dd(dd_func=_valid_pre,
                                                                        ts_x_cube=ts_x_cube,
                                                                        lbl_cube=lbl_cube)
                                nstate, nlbl = self.print_state_lbl_dd(dd_func=nxt_state.swapVariables(self.sym_vars_dict['curr_state'], self.sym_vars_dict['next_state']),
                                                                        ts_x_cube=ts_x_cube,
                                                                        lbl_cube=lbl_cube)
                                print(f"Adding edge: {cstate}{clbl} -------{action.name}------> {nstate}{nlbl}")

                            # swap variables 
                            nxt_state = nxt_state.swapVariables(self.sym_vars_dict['curr_state'], self.sym_vars_dict['next_state'])
                            
                            if layer + 1 in open_list:
                                if next_state_lbls in open_list[layer + 1]:
                                    open_list[layer + 1][next_state_lbls] |= nxt_state
                                else:
                                    open_list[layer + 1].update({next_state_lbls: nxt_state})   
                                # store the state in the correct worlf conf bucket
                                # open_list[layer + 1] |= nxt_state
                            else:
                                open_list[layer + 1] = {next_state_lbls: nxt_state}
                        
                    
            layer += 1
            
