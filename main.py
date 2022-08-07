from fileinput import filename
import os
import sys
import copy
from tabnanny import verbose

# from dd.autoref import BDD
from bidict import bidict
from src.causal_graph import CausalGraph
from src.two_player_game import TwoPlayerGame
from src.transition_system import FiniteTransitionSystem

from typing import Tuple, List
from functools import reduce

import graphviz as gv
import math
from cudd import Cudd
from itertools import count, product


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class SymbolicTransitionSystem(object):
    """
    A class to construct a symbolic transition system for each operator - in our case Action 
    """

    def __init__(self, curr_states: list , next_states: list, task, domain, manager):
        self.sym_vars_curr = curr_states
        self.sym_vars_next = next_states
        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.facts:dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        self.manager = manager
        self.actions: dict = domain.actions
        self.tr_action_idx_map: dict = {}
        self.sym_tr_actions: list = []
        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self._create_sym_var_map()
        self._initialize_bdds_for_actions()

    
    def _initialize_bdds_for_actions(self):
        """
        A function to intialize bdds for all the actions
        """
        #  initiate BDDs for all the action 
        action_idx_map = {}
        _actions = list(self.actions.keys())
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [self.manager.bddZero() for _ in range(len(self.actions))]

    
    def build_actions_tr_func(self, curr_edge_action: str):
        """
        A function to build a symbolic transition function corresponding to each action 
        """

        # flag for debugging - checking if the # of symbolic edges is same as the number of explicit edges
        added_edge: bool = False

        _actions = list(self.actions.keys())
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
        assert _action in _actions, "FIX THIS: Failed extracting a valid action."

        # for _a in _actions:
        #     if _a in _curr_action_name:
        # if _action == 'transfer':
        #     print('I am trying to transfer an object!')
        # if curr_fact in curr_edge_action.preconditions and next_fact in curr_edge_action.add_effects:
            # find it corresponding index in transtion function list 
        _idx = self.tr_action_idx_map.get(_action)
        self.sym_tr_actions[_idx] |= pre_sym & add_sym & ~del_sym
        added_edge = True
            # break
        
        # break
        
        return _action


    
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
        
        for _action in self.task.operators:
            edge_name = self.build_actions_tr_func(curr_edge_action=_action)
                
            if edge_name == 'transit':
                scheck_count_transit += 1
            elif edge_name == 'transfer':
                scheck_count_transfer += 1
            elif edge_name == 'grasp':
                scheck_count_grasp += 1
            elif edge_name == 'release':
                scheck_count_release += 1
            elif edge_name == 'human-move':
                scheck_count_hmove += 1
            else:
                print("Invalid action!!!!!")
                sys.exit(-1)


        for _action, _idx in self.tr_action_idx_map.items():
            print(f"Charateristic Function for action {_action} \n")
            print(self.sym_tr_actions[_idx], " \n")
            if plot:
                file_path = PROJECT_ROOT + f'/plots/{_action}_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/{_action}_trans_func.pdf'
                self.manager.dumpDot([self.sym_tr_actions[_idx]], file_path=file_path)
                # gv.draw('test.png')
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)
                # gv.Source.from_file(file_path).view()

        
        print("# of total symbolic transfer edge counts", scheck_count_transit)
        print("# of total symbolic transit edge counts", scheck_count_transfer)
        print("# of total symbolic grasp edge counts", scheck_count_grasp)
        print("# of total symbolic release edge counts", scheck_count_release)
        print("# of total symbolic human-move edge counts", scheck_count_hmove)
        print("All done!")
        

def create_symbolic_vars(num_of_facts: int, manager) -> Tuple[list, list]:
    """
    A helper function to create log⌈num_of_facts⌉ 
    """
    curr_state_vars: list = []
    next_state_vars: list = []

    for num_var in range(math.ceil(math.log2(num_of_facts))):
        # if num_var % 2 == 0:
            curr_state_vars.append(manager.bddVar(2*num_var, f'x{num_var}'))
            next_state_vars.append(manager.bddVar(2*num_var + 1, f'y{num_var}'))
        # else: 

    return (curr_state_vars, next_state_vars)

def create_symbolic_causal_graph(cudd_manager):
    

    _domain_file_path = PROJECT_ROOT + "/pddl_files/two_table_scenario/arch/domain.pddl"
    _problem_file_path = PROJECT_ROOT + "/pddl_files/two_table_scenario/arch/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=False)
    
    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)
    print("No. of edges in the graph:", len(_causal_graph_instance.causal_graph._graph.edges()))

    # problem = _causal_graph_instance.problem

    task_facts = _causal_graph_instance.task.facts
    task_goals = _causal_graph_instance.task.goals
    task_init = _causal_graph_instance.task.initial_state

    
    # the number of boolean variables (|x|) = log⌈|facts|⌉ - Because facts represent all possible predicates in our causal graph 
    curr_state, next_state = create_symbolic_vars(num_of_facts=len(task_facts),
                                                  manager=cudd_manager)


    return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_state, next_state, 


def get_graph(print_flag: bool = False):

    _domain_file_path = PROJECT_ROOT + + "/pddl_files/two_table_scenario/arch/domain.pddl"
    _problem_file_path = PROJECT_ROOT + + "/pddl_files/two_table_scenario/arch/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=True)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.edges())}")

    # _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    # _transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    # _transition_system_instance.build_arch_abstraction(plot=False, relabel_nodes=False)

    # if print_flag:
    #     print(f"No. of nodes in the Transition System is :"
    #           f"{len(_transition_system_instance.transition_system._graph.nodes())}")
    #     print(f"No. of edges in the Transition System is :"
    #           f"{len(_transition_system_instance.transition_system._graph.edges())}")

    # return _transition_system_instance


if __name__ == "__main__":
 

    # construct a sample rwo player game and wrap it to construct its symbolic version
    # transition_graph = get_graph(print_flag=True)
    # build_symbolic_model(transition_graph)

    cudd_manager = Cudd()
    task, domain, curr_state, next_state = create_symbolic_causal_graph(cudd_manager=cudd_manager)

    sym_tr = SymbolicTransitionSystem(curr_states=curr_state, next_states=next_state, task=task, domain=domain, manager=cudd_manager)
    sym_tr.create_transition_system(verbose=False, plot=True)