import re
import sys
import copy
import graphviz as gv

from typing import Tuple, List, Dict
from functools import reduce
from cudd import Cudd, BDD, ADD
from itertools import product


from bidict import bidict

from config import *


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

    def __init__(self, curr_states: list , next_states: list, lbl_states: list, task, domain, observations, manager):
        self.sym_vars_curr = curr_states
        self.sym_vars_next = next_states
        self.sym_vars_lbl = lbl_states
        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.facts:dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        self.domain_lbls = observations
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
        self._create_sym_state_label_map()
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
        """

        # since pre, post condition are all forzenset, we iterate over it
        pre_conds = tuple(curr_edge_action.preconditions)
        add_effects = tuple(curr_edge_action.add_effects)
        del_effects = tuple(curr_edge_action.del_effects)

        # get the bool formula for the above predicates
        pre_list = [self.predicate_sym_map_curr.get(pre_cond) for pre_cond in pre_conds]
        add_list = [self.predicate_sym_map_nxt.get(add_effect) for add_effect in add_effects]
        del_list = [self.predicate_sym_map_nxt.get(del_effect) for del_effect in del_effects]

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
        assert _action in action_list, "FIX THIS: Failed extracting a valid action."

        # pre_sym - precondition will be false when starting from the initial states; means you can take the action under all conditions
        if pre_sym.isZero():
            pre_sym = pre_sym.negate()
        
        if add_sym.isZero():
            add_sym = add_sym.negate()

        _idx = self.tr_action_idx_map.get(_action)
        self.sym_tr_actions[_idx] |= pre_sym & add_sym & ~del_sym
        
        return _action
    

    def _create_sym_state_label_map(self):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_lbl)))

        # We will add dumy brackets around the label e.g. l4 ---> (l4) becuase promela parse names the edges in that fashion
        _node_int_map_lbl = bidict({state: boolean_str[index] for index, state in enumerate(self.domain_lbls)})

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
            edge_name = self.build_actions_tr_func(curr_edge_action=_action, action_list=action_list)
            
            # if edge_name == 'transit':
            #     scheck_count_transit += 1
            # elif edge_name == 'transfer':
            #     scheck_count_transfer += 1
            # elif edge_name == 'grasp':
            #     scheck_count_grasp += 1
            # elif edge_name == 'release':
            #     scheck_count_release += 1
            # elif edge_name == 'human-move':
            #     scheck_count_hmove += 1
            # else:
            #     print("Invalid action!!!!!")
            #     sys.exit(-1)

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
            

    
    def create_state_obs_bdd(self, verbose: bool = False, plot: bool = False):
        """
        A function to create the Charactersitic function for each state observation in the domain. 
        """
        # this onlt works for grid world for now. A state looks like - `at skbn l#` and we are trying extract l# and add that as bdd
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

    def __init__(self, curr_states: List[ADD] , next_states: List[ADD], lbl_states: List[ADD], task, domain, weight_dict, observations, manager: Cudd):
        self.sym_add_vars_curr: List[ADD] = curr_states
        self.sym_add_vars_next: List[ADD] = next_states
        self.sym_add_vars_lbl: List[ADD] = lbl_states
        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.facts:dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        self.domain_lbls = observations
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
        self._create_sym_state_label_map()
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


    def _create_sym_state_label_map(self):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_add_vars_lbl)))

        _node_int_map_lbl = bidict({state: boolean_str[index] for index, state in enumerate(self.domain_lbls)})
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
    

    def create_state_obs_add(self, verbose: bool = False, plot: bool = False):
        """
        A function to create the Charactersitic function for each state observation in the domain. 
        """
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
