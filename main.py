import os
import sys
import copy
import graphviz as gv
import math

from typing import Tuple, List
from functools import reduce
from cudd import Cudd, BDD, ADD
from itertools import product


from bidict import bidict
from src.causal_graph import CausalGraph
from src.two_player_game import TwoPlayerGame
from src.transition_system import FiniteTransitionSystem
from graph_search import graph_search

from src.symbolic_search import SymbolicSearch
from src.two_player_game import TwoPlayerGame
from src.symbolic_dfa import SymbolicDFA
from src.symbolic_abstraction import SymbolicTransitionSystem


# from symbolic_planning.src.graph_search import forward_reachability

from config import *
        

def create_symbolic_vars(num_of_facts: int, manager, curr_state_var_name: str = 'x', next_state_var_name: str = 'y') -> Tuple[list, list]:
    """
    A helper function to create log⌈num_of_facts⌉
    """
    curr_state_vars: list = []
    next_state_vars: list = []

    cur_state = curr_state_var_name
    nxt_state = next_state_var_name


    for num_var in range(math.ceil(math.log2(num_of_facts))):
            curr_state_vars.append(manager.bddVar(2*num_var, f'{cur_state}{num_var}'))
            next_state_vars.append(manager.bddVar(2*num_var + 1, f'{nxt_state}{num_var}'))

    return (curr_state_vars, next_state_vars)


def create_symbolic_dfa_graph(cudd_manager, formula: str, dfa_num: int):
    _two_player_instance = TwoPlayerGame(None, None)
    _dfa = _two_player_instance.build_LTL_automaton(formula=formula, plot=False)
    _state = _dfa.get_states()

    # the number of boolean variables (|a|) = log⌈|DFA states|⌉
    curr_state, next_state = create_symbolic_vars(num_of_facts=len(_state),
                                                  manager=cudd_manager,
                                                  curr_state_var_name=f'a{dfa_num}_',
                                                  next_state_var_name=f'b{dfa_num}_')
    return curr_state, next_state, _dfa


def create_symbolic_causal_graph(cudd_manager, problem_pddl_file: str, domain_pddl_file: str):
    _causal_graph_instance = CausalGraph(problem_file=problem_pddl_file,
                                         domain_file=domain_pddl_file,
                                         draw=True)
    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)
    # print("No. of edges in the graph:", len(_causal_graph_instance.causal_graph._graph.edges()))

    task_facts = _causal_graph_instance.task.facts
    
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

    _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    _transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    # _transition_system_instance.build_arch_abstraction(plot=False, relabel_nodes=False)

    # if print_flag:
    #     print(f"No. of nodes in the Transition System is :"
    #           f"{len(_transition_system_instance.transition_system._graph.nodes())}")
    #     print(f"No. of edges in the Transition System is :"
    #           f"{len(_transition_system_instance.transition_system._graph.edges())}")

    # return _transition_system_instance


if __name__ == "__main__":
    
    BUILD_DFA: bool = True
    BUILD_ABSTRACTION: bool = False
    # construct a sample rwo player game and wrap it to construct its symbolic version
    # transition_graph = get_graph(print_flag=True)
    # build_symbolic_model(transition_graph)

    domain_file_path = PROJECT_ROOT + "/pddl_files/grid_world/domain.pddl"
    problem_file_path = PROJECT_ROOT + "/pddl_files/grid_world/problem10_10.pddl"

    cudd_manager = Cudd()

    if BUILD_DFA:
        # list of formula
        formulas = ['F(l1 & F(l2 & F(l3 & F(l4))))']
        # create a list of DFAs
        DFA_list = []

        # for now dfa_num is zero. If oyu rhave multiple DFAs then loop over them and update DFA_num
        for _idx, fmla in enumerate(formulas):
            # create boolean variables
            curr_state, next_state, _dfa = create_symbolic_dfa_graph(cudd_manager=cudd_manager, formula= fmla, dfa_num=_idx)
            # create TR corresponding to each DFA - dfa name is only used dumping graph 
            dfa_tr = SymbolicDFA(curr_states=curr_state, next_states=next_state, manager=cudd_manager, dfa=_dfa, dfa_name=f'dfa_{_idx}')
            dfa_tr.create_dfa_transition_system(verbose=True, plot=False)

        # sys.exit(-1)
    
    if BUILD_ABSTRACTION:
        task, domain, curr_state, next_state = create_symbolic_causal_graph(cudd_manager=cudd_manager,
                                                                            problem_pddl_file=problem_file_path,
                                                                            domain_pddl_file=domain_file_path)

        sym_tr = SymbolicTransitionSystem(curr_states=curr_state,
                                        next_states=next_state,
                                        task=task,
                                        domain=domain,
                                        manager=cudd_manager)
        sym_tr.create_transition_system(verbose=False, plot=False)

        print('Inital states: ', sym_tr.sym_init_states)
        print('Goal states: ', sym_tr.sym_goal_states)
        
        giant_tr = reduce(lambda a, b: a | b, sym_tr.sym_tr_actions) 

        # let do graph search from init state to a goal state
        graph_search = SymbolicSearch(init=sym_tr.sym_init_states,
                                    target=sym_tr.sym_goal_states,
                                    manager=cudd_manager,
                                    curr_vars=curr_state,
                                    next_vars=next_state,
                                    trans_func_list=sym_tr.sym_tr_actions,
                                    transition_func=giant_tr,
                                    sym_to_state=sym_tr.predicate_sym_map_curr.inv)
    
        action_list = graph_search.symbolic_bfs(verbose=False)

        # graph_search(init=sym_tr.sym_init_states,
        #              target=sym_tr.sym_goal_states,
        #              transition_func=giant_tr,
        #              x_list=curr_state,
        #              y_list=next_state)

        print("Sequence of actions")
        for _a in reversed(action_list):
            print(sym_tr.tr_action_idx_map.inv[_a])
        
        print("Done with the plan")
    
