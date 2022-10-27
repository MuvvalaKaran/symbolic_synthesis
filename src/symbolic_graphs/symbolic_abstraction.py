import re
import sys
import copy
import math 
import warnings
import graphviz as gv

from typing import Tuple, List, Dict
from functools import reduce
from cudd import Cudd, BDD, ADD
from itertools import product, zip_longest


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

    # def __init__(self, curr_states: list , next_states: list, gripper_var, on_vars, holding_vars, task, domain, manager, seg_facts: dict):
    def __init__(self, sym_vars_dict: dict, task, domain, manager, seg_facts: dict):
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
        # sort them accroding their predicates, ready, gripper status and rest will be all on stataes
        ready_state_conf = []
        gripper_state_conf = []
        on_state_conf = []   # where the boxes are placed
        # TODO: ADD a Method to sort the on predicates list 
        for istate in list(self.init) :
            if 'ready' in istate:
                ready_state_conf.append(self.predicate_sym_map_curr[istate])
            elif 'gripper' in istate:
                gripper_state_conf.append(self.predicate_sym_map_lbl[istate])
            elif 'on' in istate:
                on_state_conf.append(istate)
            else:
                warnings.warn("Error while creating the initial state. Encountered unexpect predicae. FIX THIS!!!")
                sys.exit(-1)
        
        self.sym_init_states = self.predicate_sym_map_lbl[tuple(on_state_conf)] & ready_state_conf[0] & gripper_state_conf[0]
        self.sym_goal_states = self.predicate_sym_map_lbl[tuple(self.goal)[0]] if len(self.goal) == 1 else self.predicate_sym_map_lbl[tuple(self.goal)]  # goals state will always have boxes grounded
    

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
    

    def _create_frankaworld_sym_var_map(self, task_objs: List, task_locs: List) -> dict:
        """
        A helper function used during the frankaworld construction that
            1) maps all boxes to its corresponding state label - eg. box0 - !b0 & !b1; box1 - !b0 & b1; box2 - b0 & !b1
            2) maps all locations to its corresponding state label 
        
        Note: this is not the same as state label map as a label is composition of multiple box locations and manipulatior status 
        """
        # get all the boxes and locations
        _boxes: list = task_objs  # boxes 
        _locs: list = task_locs  # locs
        _manp_status = ['gripper', 'free']

        # get the # of cprresponding boolean variables 
        num_b: int = math.ceil(math.log2(len(_boxes)))
        num_l: int = math.ceil(math.log2(len(_locs)))

        # happens when we have only one box
        if num_b == 0:
            num_b = 1

        # happens when we have only one location
        if num_l == 0:
            num_b = 1

        # create all combinations of 1-true and 0-false
        box_boolean_str = list(product([1, 0], repeat=num_b))
        loc_boolean_str = list(product([1, 0], repeat=num_l))
        manp_boolean_str = list(product([1, 0], repeat=1))

        _box_lbl_map = bidict({state: box_boolean_str[index] for index, state in enumerate(_boxes)})
        _loc_lbl_map = bidict({state: loc_boolean_str[index] for index, state in enumerate(_locs)})
        _manp_lbl_map = bidict({state: manp_boolean_str[index] for index, state in enumerate(_manp_status)})

        # loop over, create their corresponding dictionaries, combine them and return 
        counter = 0
        for d in [_box_lbl_map, _loc_lbl_map, _manp_lbl_map]:
            # loop over all the boolean string and convert them to their respective bdd vars
            for _key, _value in d.items():
                _lbl_val_list = []
                for _idx, _ele in enumerate(_value):
                    if counter == 0:
                        tmp_idx = _idx
                    elif counter == 1:
                        tmp_idx = num_b + _idx
                    else:
                        tmp_idx  = num_b + num_l + _idx
                    
                    if _ele == 1:
                        _lbl_val_list.append(self.sym_vars_lbl[tmp_idx])
                    else:
                        _lbl_val_list.append(~self.sym_vars_lbl[tmp_idx])
                
                _bool_func_curr = reduce(lambda a, b: a & b, _lbl_val_list)

                # update bidict accordingly
                d[_key] = _bool_func_curr
            counter += 1

        _manp_lbl_map.update(_box_lbl_map)
        _manp_lbl_map.update(_loc_lbl_map)
        
        # combine all the dctionary and return 
        return _manp_lbl_map


    def _get_sym_pre_conds(self, action) -> BDD:
        """
        A helper function that returns the symbolic version of the preconditions given the current action.
        """
        # get the preconditions and check is any of the current state satisfies the pre conditions
        pre_conds = tuple(action.preconditions)
        pre_list: List[BDD] = [self.monolihtic_sym_map_curr.get(pre_cond) for pre_cond in pre_conds]
        # if release take the union of pre conds as they are of same var type
        # if 'release' in action.name: 
        #     return pre_list
            # pre_sym: BDD = reduce(lambda a,b : a | b, pre_list)
        # else:
        pre_sym: BDD = reduce(lambda a,b : a & b, pre_list)

        return pre_sym
    
    def _get_sym_add_conds(self, action) -> BDD:
        """
         A helper function that returns the symbolic version of the add consitions given the current action.
        """
        add_effects = tuple(action.add_effects)
        add_list = [self.monolihtic_sym_map_nxt.get(add_effect) for add_effect in add_effects]
        add_sym: BDD = reduce(lambda a,b : a & b, add_list)

        return add_sym
    

    def _get_sym_delete_conds(self, action) -> BDD:
        """
         A helper function that returns the symbolic version of the add consitions given the current action.
        """
        del_effects = tuple(action.del_effects)

        del_list = [self.monolihtic_sym_map_nxt.get(del_effect) for del_effect in del_effects]
        if len(del_list) == 0:
            return self.manager.bddZero()
        
        del_sym = reduce(lambda a, b: a & b, del_list)

        return del_sym



    def create_transition_system_franka(self, task_objs, task_locs, verbose:bool = False, plot: bool = False):
        """
         The construction of TR for the franka world is a bit different than the gridworld. In gridworld the
          complete information of the world, i.e., the agent current location is embedded and thus
          create_trasntion_system()'s implementation sufficient when we label the states.

         Unlike gridworld, in franka world, we have to start from the iniital states with the initial conditions
          explicitly mentioned in problem file. From here, we create the state (pre) and its corresponding labels,
          we unroll the graph by taking all valid actions, compute the image, and their corresponding labels and
          keep iterating until we reach a fix point. 
        """
        # initialize 
        if verbose:
            print(f"Creating TR for Actions {self.domain.actions}")
        
        action_list = list(self.actions.keys())

        open_list = {}
        closed = self.manager.bddZero()

        init_state = self.sym_init_states
 
        ts_x_cube = reduce(lambda x, y: x & y, self.sym_vars_dict['curr_state'])
        ts_y_cube = reduce(lambda x, y: x & y, self.sym_vars_dict['next_state'])

        gripper_cube =  self.sym_vars_dict['gripper'][0]


        layer = 0
        open_list[layer] = init_state
        
        # start from the initial conditions
        while not open_list[layer].isZero():
            # remove all states that have been explored
            open_list[layer] = open_list[layer] & ~closed

            # If unexpanded states exist ... 
            if not open_list[layer].isZero():
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer]

                # compute the image of the TS states 
                for action in self.task.operators:
                    pre_sym: BDD = self._get_sym_pre_conds(action)

                    # if isinstance(pre_sym, list):
                    #     # all the precondtions should be met
                    #     _test = self.manager.bddZero() 
                    #     for p_sym in pre_sym:
                    #         _test |= open_list[layer] & p_sym
                    
                    # _valid_states = open_list[layer] & pre_sym
                    # _valid_states = pre_sym <= open_list[layer]
                    
                    # check if pre_sym is in support
                    # in_support : bool = not (open_list[layer].restrict(pre_sym) == open_list[layer])

                    #if yes, check if there is an intersection 
                    # _intersect = not (pre_sym & open_list[layer]).isZero()
                    _intersect = pre_sym & open_list[layer]
                    # if not (open_list[layer].restrict(pre_sym)) and (open_list[layer].restrict(pre_sym) == open_list[layer]):
                    if _intersect:
                        # compute the successor state and their label
                        _valid_states: BDD = open_list[layer] & pre_sym

                        # extract the labels out 
                        state_lbls = _valid_states.existAbstract(ts_x_cube)
                        
                        add_sym = self._get_sym_add_conds(action)
                        del_sym = self._get_sym_delete_conds(action)

                        nxt_state_lbls = del_sym.existAbstract(ts_y_cube) & add_sym.existAbstract(ts_y_cube)

                        if nxt_state_lbls.isOne():
                            # the box location and the gripper statis do not change, we simply carry forward the state lacel
                            # add back the state labels
                            nxt_state = ~del_sym & add_sym & state_lbls
                        else:
                            # either the box location changes or/and the gripper status
                            unchanged_lbls = state_lbls.restrict(nxt_state_lbls)
                            updated_lbls = unchanged_lbls & ~del_sym.existAbstract(ts_y_cube) & add_sym.existAbstract(ts_y_cube)
                            nxt_state = ~del_sym & add_sym & updated_lbls

                        # # extract gripper status out
                        # if not nxt_state.varIsDependent(gripper_cube):
                        #     gripper_lbl = gripper_cube if not _valid_states.restrict(gripper_cube).isZero() else ~gripper_cube
                        #     nxt_state = ~del_sym & add_sym & state_lbls & gripper_lbl
                        
                        # swap variables 
                        nxt_state = nxt_state.swapVariables(self.sym_vars_dict['curr_state'], self.sym_vars_dict['next_state'])

                        # # successor states 
                        # if del_sym is None:
                        #     # not deleting anything, just adding
                        #     succ_state = _valid_states | add_sym
                        # else:
                        #     succ_state = _valid_states.restrict(del_sym) & add_sym
                        
                        if layer + 1 in open_list:
                            open_list[layer + 1] |= nxt_state
                        else:
                            open_list[layer + 1] = nxt_state
                
                layer += 1