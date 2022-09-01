import copy
import re
import graphviz as gv

from bidict import bidict
from functools import reduce
from itertools import product

from config import *

class SymbolicDFA(object):
    """
    A class to construct a symbolic transition system for each DFA 
    """

    def __init__(self, curr_states: list , next_states: list, ts_lbls: list,predicate_sym_map_lbl: dict,  dfa, manager, dfa_name):
        self.sym_vars_curr = curr_states
        self.sym_vars_next = next_states
        self.sym_abs_lbs = ts_lbls
        self.manager = manager
        self.dfa = dfa
        self.init = dfa.get_initial_states()[0][0]
        self.goal = dfa.get_accepting_states()[0]
        self.sym_init_state = manager.bddZero()
        self.sym_goal_state = manager.bddZero()
        self.dfa_name = dfa_name
        self.dfa_bdd_tr = manager.bddZero()
        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl = predicate_sym_map_lbl
        self._create_sym_var_map()
        self._initialize_dfa_init_and_goal()
    
    def _initialize_dfa_init_and_goal(self):
        """
        Initialize symbolic init and goal states associated with DFA 
        """
        self.sym_init_state |= self.predicate_sym_map_curr.get(self.init)
        self.sym_goal_state |= self.predicate_sym_map_curr.get(self.goal)

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
        
        self.predicate_sym_map_curr = _node_int_map_curr
        self.predicate_sym_map_nxt = _node_int_map_next
    


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

    
    def get_edge_boolean_formula(self, curr_state, nxt_state):
        """
        Given an edge, extract the string and construct the boolean formula associated with this string 
        """
        # get the corresponding edge Boolean formula
        # dot.edge(f'{str(edge[0])}', f'{str(edge[1])}', label=str(edge[2].get('guard_formula')))
        
        _guard = self.dfa._graph[curr_state][nxt_state][0]['guard']
        _guard_formula = self.dfa._graph[curr_state][nxt_state][0]['guard_formula']

        symbls =  self.find_symbols(_guard_formula)

        # edge data has two attribute - guard and guard formula
        print(symbls)

        # if symbls is empty then create True edge
        if not symbls:
            return self.manager.bddOne()
        else:
            new_boolean_str = copy.deepcopy(_guard_formula)
            new_boolean_str = new_boolean_str.replace("&&", "&")
            new_boolean_str = new_boolean_str.replace("||", "|")
            new_boolean_str = new_boolean_str.replace("!", "~")

            # get the the corresponding boolean formula
            sym_edge_formula = self.predicate_sym_map_lbl.get(new_boolean_str)

            if sym_edge_formula is None:
                sym_edge_formula = self.manager.bddZero()

            # # iterate over the symbolc and replace them with symbolic boolean representation
            # for symbol in symbls:
            #     assert self.predicate_sym_map_lbl[symbol] is not None, "Trying to access a Boolean formula of unknown observation label"
            #     new_boolean_str = new_boolean_str.replace(symbol, self.predicate_sym_map_lbl[symbol])

            return sym_edge_formula
    

    def create_dfa_transition_system(self, verbose: bool = False, plot: bool = False):
        """
        A function to create the TR function for each Action we hav defined in the domain
        """
        for _curr, _nxt in self.dfa._graph.edges():
            # get the boolean formula for the corresponding edge 
            _curr_sym = self.predicate_sym_map_curr.get(_curr) 
            _nxt_sym = self.predicate_sym_map_nxt.get(_nxt)
            _edge_sym =  self.get_edge_boolean_formula(curr_state=_curr, nxt_state=_nxt)
           
            
            self.dfa_bdd_tr |= _curr_sym & _nxt_sym & _edge_sym
    
        if verbose:
            print(f"Charateristic Function for DFA  is \n")
            print(self.dfa_bdd_tr, " \n")
            if plot:
                file_path = PROJECT_ROOT + f'/plots/{self.dfa_name}_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/{self.dfa_name}_trans_func.pdf'
                self.manager.dumpDot([self.dfa_bdd_tr], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)