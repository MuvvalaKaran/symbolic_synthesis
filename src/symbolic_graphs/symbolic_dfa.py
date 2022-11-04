import re
import sys
import copy
import warnings
import graphviz as gv

from typing import List, Union
from bidict import bidict
from functools import reduce
from itertools import product

from cudd import Cudd, BDD, ADD

from regret_synthesis_toolbox.src.graph import DFAGraph

from utls import *
from config import *


class SymbolicDFA(object):
    """
    A class to construct a symbolic transition system for each DFA 
    """

    def __init__(self,
                 curr_states: List[BDD],
                 next_states: List[BDD],
                 predicate_sym_map_lbl: dict,
                 dfa: DFAGraph,
                 manager: Cudd,
                 dfa_name,
                 ltlf_flag: bool = False):
        self.sym_vars_curr: List[BDD] = curr_states
        self.sym_vars_next: List[BDD] = next_states
        self.manager: Cudd = manager
        self.dfa = dfa
        self.sym_init_state: BDD = manager.bddZero()
        self.sym_goal_state: BDD = manager.bddZero()
        self.dfa_name: str = dfa_name
        self.dfa_bdd_tr: BDD = manager.bddZero()
        self.dfa_predicate_sym_map_curr: bidict = {}
        self.dfa_predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl = predicate_sym_map_lbl

        if ltlf_flag:
            self.init: List[int] = dfa.init_state
            self.goal: List[int] = dfa.accp_states
        else:
            self.init: str = dfa.get_initial_states()[0][0]
            self.goal: str = dfa.get_accepting_states()[0]

        self._create_sym_var_map()
        self._initialize_dfa_init_and_goal(ltlf_flag=ltlf_flag)
    
    def _initialize_dfa_init_and_goal(self, ltlf_flag: bool = False):
        """
        Initialize symbolic init and goal states associated with DFA 
        """
        # ltlf formulas can have multiple accepting states
        if ltlf_flag:
            for i_st in self.init:
                self.sym_init_state |= self.dfa_predicate_sym_map_curr.get(i_st)

            for g_st in self.goal:
                self.sym_goal_state |= self.dfa_predicate_sym_map_curr.get(g_st)
        else:
            self.sym_init_state |= self.dfa_predicate_sym_map_curr.get(self.init)
            self.sym_goal_state |= self.dfa_predicate_sym_map_curr.get(self.goal)

        assert self.sym_init_state.isZero() is False and self.sym_goal_state.isZero() is False, \
        "Couldn't build the symbolic init and goal states of DFA. FIX THIS!!!"


    def _create_sym_var_map(self):
        """
        Loop through all the States that are reachable and assign a boolean funtion to it
        """

        # for dfa in self.dfa_list:
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_curr)))
        
        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(list(self.dfa._graph.nodes()))})
        _node_int_map_next = copy.deepcopy(_node_int_map_curr)

        assert len(boolean_str) >= len(_node_int_map_next), "FIX THIS: Looks like there are more Facts that boolean variables!"

        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map_curr.items():
            _curr_val_list = []
            _next_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
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
        
        self.dfa_predicate_sym_map_curr = _node_int_map_curr
        self.dfa_predicate_sym_map_nxt = _node_int_map_next

    
    def in_order_nnf_tree_traversal(self, expression, formula):
        """
        Traverse the edge formula given by Promela a binary tree. This function implements a in-order tree traversal algorithm.
        """
        if hasattr(formula, 'symbol'):
            # get the corresponding boolean expression
            if '!' in formula.name:
                return ~self.predicate_sym_map_lbl.get(formula.symbol)
            else:
                return self.predicate_sym_map_lbl.get(formula.symbol)
        
        expression = self.in_order_nnf_tree_traversal(expression, formula.left)
        if formula.name == 'AND':
            expression = expression & self.in_order_nnf_tree_traversal(expression, formula.right)
        elif formula.name == 'OR':
            expression |= self.in_order_nnf_tree_traversal(expression, formula.right)

        return expression


    def find_symbols(self, formula):
        """
        Find symbols associated with an edge
        """
        regex = re.compile(r"[a-z]+[a-z0-9]*")
        matches = regex.findall(formula)
        symbols = list()
        for match in matches:
            symbols += [match]
        symbols = list(set(symbols))
        symbols.sort()
        return symbols

    
    def get_edge_boolean_formula(self, curr_state, nxt_state, valid_dfa_edge_formula_size: int = 1):
        """
        Given an edge, extract the string and construct the boolean formula associated with this string 
        """        
        _guard = self.dfa._graph[curr_state][nxt_state][0]['guard']
        _guard_formula = self.dfa._graph[curr_state][nxt_state][0]['guard_formula']

        symbls =  self.find_symbols(_guard_formula)

        # if symbls is empty then create True edge
        if not symbls or 'true' in symbls:
            return self.manager.bddOne()
        else:
            if len(symbls) > valid_dfa_edge_formula_size:
                return self.manager.bddZero()

            elif len(symbls) <= valid_dfa_edge_formula_size:
                # for gird world, each state has only one map. But, the edges on the DFA for formula like F(l1 & F(l2))
                #  will have edges like ((l1)&(!(l2))) and ((l1)&(l2)). While the later is not physically possbile, the first is umabiguous way of
                #  expressing only 1 symbol. Thus, its a valid edge and we need to create a corresponding boolean formula
                edgy_formula = self.in_order_nnf_tree_traversal(expression=self.manager.bddZero() , formula=_guard)
                return edgy_formula


    def create_dfa_transition_system(self, verbose: bool = False, plot: bool = False, valid_dfa_edge_formula_size: int = 1):
        """
        A function to create the TR function for each valid transition in a DFA.
        """
        for _curr, _nxt in self.dfa._graph.edges():
            # get the boolean formula for the corresponding edge 
            _curr_sym = self.dfa_predicate_sym_map_curr.get(_curr) 
            _nxt_sym = self.dfa_predicate_sym_map_nxt.get(_nxt)
            _edge_sym = self.get_edge_boolean_formula(curr_state=_curr,
                                                      nxt_state=_nxt,
                                                      valid_dfa_edge_formula_size=valid_dfa_edge_formula_size)
            
            if not isinstance(_edge_sym, BDD):
                _edge = self.dfa._graph[_curr][_nxt][0]['guard_formula']
                warnings.warn(f"Error while parsing the LTL Formula. Could not parse edge {_edge}")
                sys.exit(-1)
            
            self.dfa_bdd_tr |= _curr_sym & _nxt_sym & _edge_sym
    
        if verbose:
            print(f"Charateristic Function for DFA  is \n")
            print(self.dfa_bdd_tr, " \n")
            if plot:
                file_path = PROJECT_ROOT + f'/plots/{self.dfa_name}_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/{self.dfa_name}_trans_func.pdf'
                self.manager.dumpDot([self.dfa_bdd_tr], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)
    

    def get_ltlf_edge_boolean_formula(self, labels: List, guard: str) -> BDD:
        """
        A function that parse the guard and constructs its correpsonding symbolic edge for symbolic LTLf DFA construction
        """
        expr = self.manager.bddOne()

        for idx, value in enumerate(guard):
            if value == "1":
                expr = expr & self.predicate_sym_map_lbl.get(str(labels[idx]) if isinstance(labels, tuple) else str(labels))
            elif value == "0":
                expr = expr & ~self.predicate_sym_map_lbl.get(str(labels[idx]) if isinstance(labels, tuple) else str(labels))
            else:
                assert value == "X", "Error while constructing symbolic LTLF DAF edge. FIX THIS!!!"
        
        return expr
    

    def create_symbolic_ltlf_transition_system(self, verbose: bool = False, plot: bool = False):
        """
        This function parses the Mona DFA output and construct the symbolic TR associated with DFA.
        """

        mona_output: str = self.dfa.mona_dfa

        for line in mona_output.splitlines():
            if line.startswith("State "):
                # extract the original state
                orig_state = self.dfa.get_value(line, r".*State[\s]*(\d+):\s.*", int)
                
                # extract string guard
                guard = self.dfa.get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                
                # convert it into boolean formula
                if self.dfa.task_labels:
                    _edge_sym = self.get_ltlf_edge_boolean_formula(self.dfa.task_labels, guard)
                else:
                    _edge_sym = self.get_ltlf_edge_boolean_formula(self.dfa.task_labels, "X")
                
                dest_state = self.dfa.get_value(line, r".*state[\s]*(\d+)[\s]*.*", int)

                # ignore the superficial state 0
                if orig_state:
                    _curr_sym = self.dfa_predicate_sym_map_curr.get(orig_state) 
                    _nxt_sym = self.dfa_predicate_sym_map_nxt.get(dest_state)
                    self.dfa_bdd_tr |= _curr_sym & _nxt_sym & _edge_sym
        
        if verbose:
            print(f"Charateristic Function for DFA  is \n")
            print(self.dfa_bdd_tr, " \n")
            if plot:
                file_path = PROJECT_ROOT + f'/plots/{self.dfa_name}_ADD_ltlf_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/{self.dfa_name}_ADD_ltlf_trans_func.pdf'
                self.manager.dumpDot([self.dfa_bdd_tr], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)       



class SymbolicAddDFA(object):
    """
    A class to construct a symbolic transition system for each DFA 
    """

    def __init__(self,
                curr_states: List[ADD],
                next_states: List[ADD],
                predicate_add_sym_map_lbl: dict,
                predicate_sym_map_lbl: dict,
                dfa: DFAGraph,
                manager: Cudd,
                dfa_name: str,
                ltlf_flag: bool = False):
        self.sym_add_vars_curr: List[ADD] = curr_states
        self.sym_add_vars_next: List[ADD] = next_states
        self.manager: Cudd = manager
        self.dfa = dfa
        self.sym_init_state: ADD = manager.addZero()
        self.sym_goal_state: ADD = manager.addZero()
        self.dfa_name: str = dfa_name
        self.dfa_bdd_tr: ADD = manager.addZero()
        
        self.dfa_predicate_add_sym_map_curr: bidict = {}
        self.dfa_predicate_add_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl = predicate_sym_map_lbl

        self.dfa_predicate_sym_map_curr: bidict = {}
        self.dfa_predicate_sym_map_nxt: bidict = {}
        self.predicate_add_sym_map_lbl = predicate_add_sym_map_lbl

        if ltlf_flag:
            self.init: List[int] = dfa.init_state
            self.goal: List[int] = dfa.accp_states
        else:
            self.init: str = dfa.get_initial_states()[0][0]
            self.goal: str = dfa.get_accepting_states()[0]
        
        self._create_sym_var_map()
        self._initialize_dfa_init_and_goal(ltlf_flag=ltlf_flag)
    
    def _initialize_dfa_init_and_goal(self, ltlf_flag: bool = False):
        """
        Initialize symbolic init and goal states associated with DFA 
        """        
        # ltlf formulas can have multiple accepting states
        if ltlf_flag:
            for i_st in self.init:
                self.sym_init_state |= self.dfa_predicate_add_sym_map_curr.get(i_st)

            for g_st in self.goal:
                self.sym_goal_state |= self.dfa_predicate_add_sym_map_curr.get(g_st)

        else:
            self.sym_init_state |= self.dfa_predicate_add_sym_map_curr.get(self.init)
            self.sym_goal_state |= self.dfa_predicate_add_sym_map_curr.get(self.goal)

        assert self.sym_init_state.isZero() is False and self.sym_goal_state.isZero() is False, \
        "Couldn't build the symbolic init and goal states of DFA. FIX THIS!!!"


    def _create_sym_var_map(self):
        """
        Loop through all the States that are reachable and assign a boolean funtion to it
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_add_vars_curr)))
        
        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(list(self.dfa._graph.nodes()))})
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
            
            _bool_add_func_curr = reduce(lambda a, b: a & b, _curr_val_list)
            _bool_add_func_nxt = reduce(lambda a, b: a & b, _next_val_list)

            _bool_bdd_func_curr = _bool_add_func_curr.bddPattern()
            _bool_bdd_func_nxt = _bool_add_func_nxt.bddPattern()

            # update bidict accordingly
            _node_int_map_curr[_key] = _bool_add_func_curr
            _node_int_map_next[_key] = _bool_add_func_nxt    

            _node_bdd_int_map_curr[_key] = _bool_bdd_func_curr
            _node_bdd_int_map_next[_key] = _bool_bdd_func_nxt 
        
        self.dfa_predicate_add_sym_map_curr = _node_int_map_curr
        self.dfa_predicate_add_sym_map_nxt = _node_int_map_next

        self.dfa_predicate_sym_map_curr = _node_bdd_int_map_curr
        self.dfa_predicate_sym_map_nxt = _node_bdd_int_map_next
    
    def in_order_nnf_tree_traversal(self, expression, formula):
        """
        Traverse the edge formula given by Promela a binary tree. This function implements a in-order tree traversal algorithm.
        """
        if hasattr(formula, 'symbol'):
            # get the corresponding boolean expression
            if '!' in formula.name:
                return ~self.predicate_add_sym_map_lbl.get(formula.symbol)
            else:
                return self.predicate_add_sym_map_lbl.get(formula.symbol)
        
        expression = self.in_order_nnf_tree_traversal(expression, formula.left)
        if formula.name == 'AND':
            expression = expression & self.in_order_nnf_tree_traversal(expression, formula.right)
        elif formula.name == 'OR':
            expression |= self.in_order_nnf_tree_traversal(expression, formula.right)

        return expression


    def find_symbols(self, formula: str):
        """
        Find symbols associated with an edge
        """
        regex = re.compile(r"[a-z]+[a-z0-9]*")
        matches = regex.findall(formula)
        symbols = list()
        for match in matches:
            symbols += [match]
        symbols = list(set(symbols))
        symbols.sort()
        return symbols

    
    def get_edge_boolean_formula(self, curr_state, nxt_state, valid_dfa_edge_formula_size: int = 1):
        """
        Given an edge, extract the string and construct the boolean formula associated with this string 
        """        
        _guard = self.dfa._graph[curr_state][nxt_state][0]['guard']
        _guard_formula = self.dfa._graph[curr_state][nxt_state][0]['guard_formula']

        symbls =  self.find_symbols(_guard_formula)

        # if symbls is empty then create True edge
        if not symbls or 'true' in symbls:
            return self.manager.addOne()
        else:
            if len(symbls) > valid_dfa_edge_formula_size:
                return self.manager.addZero()

            elif len(symbls) <= valid_dfa_edge_formula_size:
                # for gird world, each state has only one map. But, the edges on the DFA for formula like F(l1 & F(l2))
                #  will have edges like ((l1)&(!(l2))) and ((l1)&(l2)). While the later is not physically possbile, the first is umabiguous way of
                #  expressing only 1 symbol. Thus, its a valid edge and we need to create a corresponding boolean formula
                edgy_formula = self.in_order_nnf_tree_traversal(expression=self.manager.addZero() , formula=_guard)
                return edgy_formula

    

    def create_dfa_transition_system(self, verbose: bool = False, plot: bool = False, valid_dfa_edge_formula_size: int = 1):
        """
        A function to create the TR function for each valid transition in a DFA.
        """
        for _curr, _nxt in self.dfa._graph.edges():
            # get the boolean formula for the corresponding edge 
            _curr_sym = self.dfa_predicate_add_sym_map_curr.get(_curr) 
            _nxt_sym = self.dfa_predicate_add_sym_map_nxt.get(_nxt)
            _edge_sym = self.get_edge_boolean_formula(curr_state=_curr,
                                                      nxt_state=_nxt,
                                                      valid_dfa_edge_formula_size=valid_dfa_edge_formula_size)
            
            self.dfa_bdd_tr |= _curr_sym & _nxt_sym & _edge_sym
    
        if verbose:
            print(f"Charateristic Function for DFA  is \n")
            print(self.dfa_bdd_tr, " \n")
            if plot:
                file_path = PROJECT_ROOT + f'/plots/{self.dfa_name}_ADD_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/{self.dfa_name}_ADD_trans_func.pdf'
                self.manager.dumpDot([self.dfa_bdd_tr], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)
    

    def get_ltlf_edge_boolean_formula(self, labels: List, guard: str) -> ADD:
        """
        A function that parse the guard and constructs its correpsonding symbolic edge for symbolic LTLf DFA construction
        """
        expr = self.manager.addOne()

        for idx, value in enumerate(guard):
            if value == "1":
                expr = expr & self.predicate_add_sym_map_lbl.get(str(labels[idx]) if isinstance(labels, tuple) else str(labels))
            elif value == "0":
                expr = expr & ~self.predicate_add_sym_map_lbl.get(str(labels[idx]) if isinstance(labels, tuple) else str(labels))
            else:
                assert value == "X", "Error while constructing symbolic LTLF DAF edge. FIX THIS!!!"
        
        return expr
    

    def create_symbolic_ltlf_transition_system(self, verbose: bool = False, plot: bool = False):
        """
        This function parses the Mona DFA output and construct the symbolic TR associated with DFA.
        """

        mona_output: str = self.dfa.mona_dfa

        for line in mona_output.splitlines():
            if line.startswith("State "):
                # extract the original state
                orig_state = self.dfa.get_value(line, r".*State[\s]*(\d+):\s.*", int)
                
                # extract string guard
                guard = self.dfa.get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                
                # convert it into boolean formula
                if self.dfa.task_labels:
                    _edge_sym = self.get_ltlf_edge_boolean_formula(self.dfa.task_labels, guard)
                else:
                    _edge_sym = self.get_ltlf_edge_boolean_formula(self.dfa.task_labels, "X")
                
                dest_state = self.dfa.get_value(line, r".*state[\s]*(\d+)[\s]*.*", int)

                # ignore the superficial state 0
                if orig_state:
                    _curr_sym = self.dfa_predicate_add_sym_map_curr.get(orig_state) 
                    _nxt_sym = self.dfa_predicate_add_sym_map_nxt.get(dest_state)
                    self.dfa_bdd_tr |= _curr_sym & _nxt_sym & _edge_sym
        
        if verbose:
            print(f"Charateristic Function for DFA  is \n")
            print(self.dfa_bdd_tr, " \n")
            if plot:
                file_path = PROJECT_ROOT + f'/plots/{self.dfa_name}_ADD_ltlf_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/{self.dfa_name}_ADD_ltlf_trans_func.pdf'
                self.manager.dumpDot([self.dfa_bdd_tr], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)


class SymbolicDFAFranka(SymbolicDFA):
    """
     Class that inherits Symbolic DFA and construct symbolic Transition Relation associated with Formulas for Franka Abstraction
    """
    def __init__(self,
                 curr_states: List[BDD],
                 next_states: List[BDD],
                 predicate_sym_map_lbl: dict,
                 dfa: DFAGraph,
                 manager: Cudd,
                 dfa_name,
                 pred_int_map: dict,
                 ltlf_flag: bool = False):

        super().__init__(curr_states, next_states, predicate_sym_map_lbl, dfa, manager, dfa_name, ltlf_flag)
        self.pred_int_map = pred_int_map


    def get_ltlf_edge_boolean_formula(self, labels: List, guard: str) -> BDD:
        """
         This function overirdes the base function by parsing the atomic proposition of form p00 to its corresponding wordl conf.

         p00 := the first int corresponds to box id and the second int corresponds to box's location. 
         Thus, this proposition says box0 should be at loc l0. 
        """
        expr = self.manager.bddOne()
        for idx, value in enumerate(guard):
            if value == "1":
                if isinstance(labels, tuple):
                    cryptic_lbl = labels[idx]
                else:
                    cryptic_lbl = labels
                
                # get the gripper free predicate if label is `free` and high
                if 'free' in str(cryptic_lbl):
                    expr = expr & self.predicate_sym_map_lbl['(gripper free)']
                else:
                    # extract the box id and loc as str
                    box_loc: str = re.search(r'\d+', str(cryptic_lbl)).group()
                    expr = expr & self.predicate_sym_map_lbl[f'(on b{box_loc[0]} l{box_loc[1]})']
            
            elif value == "0":
                if isinstance(labels, tuple):
                    cryptic_lbl = labels[idx]
                else:
                    cryptic_lbl = labels
                
                 # get the not(gripper free) predicate if label is `free` and low
                if 'free' in str(cryptic_lbl):
                    expr = expr & ~self.predicate_sym_map_lbl['(gripper free)']
                else:
                    # extract the box id and loc as str
                    box_loc: str = re.search(r'\d+', str(cryptic_lbl)).group()
                    expr = expr & ~self.predicate_sym_map_lbl[f'(on b{box_loc[0]} l{box_loc[1]})']
            else:
                assert value == "X", "Error while constructing symbolic LTLF DAF edge. FIX THIS!!!"
        
        return expr


    def in_order_nnf_tree_traversal(self, expression, formula):
        """
        Traverse the edge formula given by Promela a binary tree. This function implements a in-order tree traversal algorithm.
        """
        if hasattr(formula, 'symbol'):
            # get the corresponding boolean expression
            if '!' in formula.name:
                if 'free' in formula.name:
                    return ~self.predicate_sym_map_lbl['(gripper free)']
                else:
                    # extract the box id and loc as str
                    box_loc: str = re.search(r'\d+', formula.name).group()
                    return ~self.predicate_sym_map_lbl[f'(on b{box_loc[0]} l{box_loc[1]})']
            else:
                if 'free' in formula.name:
                    return self.predicate_sym_map_lbl['(gripper free)']
                else:
                    # extract the box id and loc as str
                    box_loc: str = re.search(r'\d+', formula.name).group()
                    return self.predicate_sym_map_lbl[f'(on b{box_loc[0]} l{box_loc[1]})']
        
        expression = self.in_order_nnf_tree_traversal(expression, formula.left)
        if formula.name == 'AND':
            expression = expression & self.in_order_nnf_tree_traversal(expression, formula.right)
        elif formula.name == 'OR':
            expression |= self.in_order_nnf_tree_traversal(expression, formula.right)

        return expression
    

class SymbolicAddDFAFranka(SymbolicAddDFA):
    """
     Class that inherits Symbolic ADD DFA and constructs symbolic Transition Relation associated with Formulas for Franka Abstraction
    """

    def __init__(self,
                 curr_states: List[ADD],
                 next_states: List[ADD],
                 predicate_add_sym_map_lbl: dict,
                 predicate_sym_map_lbl: dict,
                 dfa: DFAGraph, manager: Cudd,
                 dfa_name: str,
                 pred_int_map: dict,
                 ltlf_flag: bool = False):
        super().__init__(curr_states, next_states, predicate_add_sym_map_lbl, predicate_sym_map_lbl, dfa, manager, dfa_name, ltlf_flag)
        self.pred_int_map = pred_int_map

    
    def get_ltlf_edge_boolean_formula(self, labels: List, guard: str) -> ADD:
        """
         Construct the edge formulas after parsing the atomic proposition and looking up its corresponsind ADD repr
        """
        
        expr = self.manager.addOne()
        for idx, value in enumerate(guard):
            if value == "1":
                if isinstance(labels, tuple):
                    cryptic_lbl = labels[idx]
                else:
                    cryptic_lbl = labels
                
                # get the gripper free predicate if label is `free` and high
                if 'free' in str(cryptic_lbl):
                    expr = expr & self.predicate_add_sym_map_lbl['(gripper free)']
                else:
                    # extract the box id and loc as str
                    box_loc: str = re.search(r'\d+', str(cryptic_lbl)).group()
                    expr = expr & self.predicate_add_sym_map_lbl[f'(on b{box_loc[0]} l{box_loc[1]})']
            
            elif value == "0":
                if isinstance(labels, tuple):
                    cryptic_lbl = labels[idx]
                else:
                    cryptic_lbl = labels
                
                 # get the not(gripper free) predicate if label is `free` and low
                if 'free' in str(cryptic_lbl):
                    expr = expr & ~self.predicate_add_sym_map_lbl['(gripper free)']
                else:
                    # extract the box id and loc as str
                    box_loc: str = re.search(r'\d+', str(cryptic_lbl)).group()
                    expr = expr & ~self.predicate_add_sym_map_lbl[f'(on b{box_loc[0]} l{box_loc[1]})']
            else:
                assert value == "X", "Error while constructing symbolic LTLF DAF edge. FIX THIS!!!"
        
        return expr