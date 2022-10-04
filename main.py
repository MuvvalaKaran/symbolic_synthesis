import sys
import time
import math

from typing import Tuple, List
from cudd import Cudd, BDD, ADD
from itertools import product


from src.explicit_graphs import CausalGraph
from src.explicit_graphs import TwoPlayerGame
from src.explicit_graphs import FiniteTransitionSystem

from src.symbolic_graphs import SymbolicDFA, SymbolicAddDFA, SymbolicMultipleDFA, SymbolicMultipleAddDFA
from src.symbolic_graphs import SymbolicTransitionSystem, SymbolicWeightedTransitionSystem

from src.algorithms.blind_search import SymbolicSearch
from src.algorithms.monolithic_search.blind_search import MultipleFormulaBFS
from src.algorithms.weighted_search import SymbolicDijkstraSearch
# from src.algorithms.old_disjoint_search import MultipleFormulaDijkstra

from src.simulate_strategy import create_gridworld, \
     convert_action_dict_to_gridworld_strategy, plot_policy, convert_action_dict_to_gridworld_strategy_nLTL


from utls import get_graph

import  src.gridworld_visualizer.gridworld_vis.gridworld as gridworld_handle


from config import *


def set_variable_reordering(manager: Cudd, make_tree_node: bool = False, verbose: bool = False, **kwargs):
    """
    This function is called if variable reordering is set to true.

    Different ways to speed up the process
    1. AutodynaEnable() - Enable Dyanmic vairable reordering
    2. ReorderingStatus() - Return the current reordering status and default method
    3. EnablingOrderingMonitoring() - Enable monitoring of a variaable order 
    4. maxReorderings() - Read and set maximum number of variable reorderings 
    5. EnablereorderingReport() - Enable reporting of variable reordering

    MakeTreeNode() - allows us to specify constraints over groups of variables. For example, we can that x, x'
    need to always be contiguous. Thus the relative ordering within the group is left unchanged. 

    MTR takes in two args -
     low: 
     size: 2 (grouping curr state vars and their corresponding primes together)

    """
    manager.autodynEnable()

    if make_tree_node:
        # Current, we follow the convention where we first build the TS variables, then the observations,
        # and finally the dfa variables. Within the TS and DFA, we pait vars and their primes together.
        # The observation variables are all grouped together as one.
        var_reorder_counter = 0  
        for i in range(manager.size()):
        # for i in range(kwargs['ts_sym_var_len']):
            if i<= kwargs['ts_sym_var_len']:
                manager.makeTreeNode(2*i, 2)
            # elif i > kwargs['ts_sym_var_len'] and i <= kwargs['ts_obs_var_len']:
            #     manager.makeTreeNode(kwargs['ts_obs_var_len']*i, kwargs['ts_obs_var_len'])
            # elif i> kwargs['ts_obs_var_len'] + kwargs['ts_sym_var_len']:
            #     manager.makeTreeNode(2*i, 2)

    if verbose:
        manager.enableOrderingMonitoring()
    else:
        manager.enableReorderingReporting()



def create_symbolic_lbl_vars(lbls,
                             manager: Cudd,
                             label_state_var_name: str = 'l',
                             valid_dfa_edge_symbol_size: int = 1,
                             add_flag: bool = False):
    """
    This function create boolean vairables used to create observation labels for each state. Note that in this method we do create
    prime variables as they do not switch their values.

    The number of variable you needs to represent the state observation depends on two factors.
     1. The number of objects (PDDL problem file terminology) you have (informally locations)
     2. The number of locations your observations your state observation contains
    
    e.g. Grid World: The observation associated with each state is only the location of that cell.
     So, you cannot observe l2 and l1 at the same time as our robot can only be in one cell at a given time. 

     Thus, the set of valid observation is {l1, l2}
    
    For Franka World: Say, the observation consists of the location of the robot and an object. The set of Valid locations is
     {else, l1, l2}. The set of valid observation is {(else, else), (else, l1), (else, l2), (l1, else), (l1, l1),
     (l1, l2), (l2, else) (l2, l1), (l2, l2)}

    """
    # We modify the labels so that they are inline with promela's edge labelling format in DFA.
    #  The atomic propositions are always of the form (l2) or (!(l2))

    # remove skbn form the lbls as it contributes to unnecessary variables space
    try:
        del lbls['skbn']
    except KeyError:
        print("Looks like the label 'skbn' is not an object in the PDDL problem file.")


    possible_obs: list = []
    for ele in lbls:
        possible_obs.append(ele)
    # possible_obs = lbls  # for Frank world you have to update this
    if valid_dfa_edge_symbol_size > 1:
        possible_obs = list(product(possible_obs, repeat=valid_dfa_edge_symbol_size))

    state_lbl_vars: list = []
    lbl_state = label_state_var_name

    # get the number of variables in the manager. We will assign the next idex to the next lbl variables
    _num_of_sym_vars = cudd_manager.size()

    # NOTE: This might cause issue in the future. len - 2 is hack. You should actually remove the irrelevant objects from objs 
    num_of_lbls = len(possible_obs)
    for num_var in range(math.ceil(math.log2(num_of_lbls))):
        _var_index = num_var + _num_of_sym_vars
        if add_flag:
            state_lbl_vars.append(manager.addVar(_var_index, f'{lbl_state}{num_var}'))
        else:
            state_lbl_vars.append(manager.bddVar(_var_index, f'{lbl_state}{num_var}'))

    return state_lbl_vars, possible_obs

    
def create_symbolic_vars(num_of_facts: int,
                         manager: Cudd,
                         curr_state_var_name: str = 'x',
                         next_state_var_name: str = 'y',
                         add_flag: bool = False) -> Tuple[list, list]:
    """
    A helper function to create log⌈num_of_facts⌉
    """
    curr_state_vars: list = []
    next_state_vars: list = []

    cur_state = curr_state_var_name
    nxt_state = next_state_var_name

    # get the number of variables in the manager. We will assign the next idex to the next lbl variables
    _num_of_sym_vars = manager.size()

    for num_var in range(math.ceil(math.log2(num_of_facts))):
            if add_flag:
                curr_state_vars.append(manager.addVar(_num_of_sym_vars + (2*num_var), f'{cur_state}{num_var}'))
                next_state_vars.append(manager.addVar(_num_of_sym_vars + (2*num_var + 1), f'{nxt_state}{num_var}'))
            else:
                curr_state_vars.append(manager.bddVar(_num_of_sym_vars + (2*num_var), f'{cur_state}{num_var}'))
                next_state_vars.append(manager.bddVar(_num_of_sym_vars + (2*num_var + 1), f'{nxt_state}{num_var}'))

    return (curr_state_vars, next_state_vars)


def create_symbolic_dfa_graph(cudd_manager,
                              formula: str,
                              dfa_num: int,
                              add_flag: bool = False):
    _two_player_instance = TwoPlayerGame(None, None)
    _dfa = _two_player_instance.build_LTL_automaton(formula=formula, plot=False)
    _state = _dfa.get_states()

    # the number of boolean variables (|a|) = log⌈|DFA states|⌉
    curr_state, next_state = create_symbolic_vars(num_of_facts=len(_state),
                                                  manager=cudd_manager,
                                                  curr_state_var_name=f'a{dfa_num}_',
                                                  next_state_var_name=f'b{dfa_num}_',
                                                  add_flag=add_flag)
    return curr_state, next_state, _dfa


def create_symbolic_causal_graph(cudd_manager,
                                 problem_pddl_file: str,
                                 domain_pddl_file: str,
                                 create_lbl_vars: bool,
                                 draw_causal_graph: bool = False,
                                 max_valid_formula_size: int = 1,
                                 add_flag: bool = False):
    """
    A function to create an instance of causal graph which call pyperplan. We access the task related properties pyperplan
     and create symbolic TR related to action.   
    """
    _causal_graph_instance = CausalGraph(problem_file=problem_pddl_file,
                                         domain_file=domain_pddl_file,
                                         draw=draw_causal_graph)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)
    # print("No. of edges in the graph:", len(_causal_graph_instance.causal_graph._graph.edges()))

    task_facts = _causal_graph_instance.task.facts
    
    # the number of boolean variables (|x|) = log⌈|facts|⌉ - Because facts represent all possible predicates in our causal graph 
    curr_state, next_state = create_symbolic_vars(num_of_facts=len(task_facts),
                                                  manager=cudd_manager,
                                                  add_flag=add_flag)
                                                  
    if create_lbl_vars:
        print("*****************Creating Boolean variables for Labels as well! This functionality only works for grid world!*****************")
        objs = _causal_graph_instance.problem.objects
        
        lbl_state, possible_obs = create_symbolic_lbl_vars(lbls=objs,
                                                           manager=cudd_manager,
                                                           valid_dfa_edge_symbol_size=max_valid_formula_size,
                                                           add_flag=add_flag)

        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, possible_obs, \
         curr_state, next_state, lbl_state

    return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_state, next_state



def build_add_symbolic_dfa(formulas: List[str],
                           sym_tr_handle: SymbolicWeightedTransitionSystem,
                           manager: Cudd,
                           verbose: bool = False,
                           plot: bool = False) -> Tuple[List[SymbolicDFA], List[ADD], List[ADD]]:
    """
    A helper function to build a symbolic DFA given a formula from ADD Variables.
    """
    
    # create a list of DFAs
    DFA_handles = []
    DFA_curr_vars = []
    DFA_nxt_vars = []

    for _idx, fmla in enumerate(formulas):
        # create different ADD variables for different DFAs
        add_dfa_curr_state, add_dfa_next_state, _dfa = create_symbolic_dfa_graph(cudd_manager=cudd_manager,
                                                                                 formula= fmla,
                                                                                 dfa_num=_idx,
                                                                                 add_flag=True)

        # create TR corresponding to each DFA - dfa name is only used dumping graph 
        dfa_tr = SymbolicAddDFA(curr_states=add_dfa_curr_state,
                                next_states=add_dfa_next_state,
                                predicate_add_sym_map_lbl=sym_tr_handle.predicate_add_sym_map_lbl,
                                predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                manager=manager,
                                dfa=_dfa,
                                dfa_name=f'dfa_{_idx}')

        dfa_tr.create_dfa_transition_system(verbose=verbose,
                                            plot=plot,
                                            valid_dfa_edge_formula_size=len(_dfa.get_symbols()))

        # We extend DFA vars list as we dont need them in stored in separate lists
        DFA_handles.append(dfa_tr)
        DFA_curr_vars.extend(add_dfa_curr_state)
        DFA_nxt_vars.extend(add_dfa_next_state)
    
    return DFA_handles, DFA_curr_vars, DFA_nxt_vars



def build_bdd_symbolic_dfa(formulas: List[str],
                           sym_tr_handle: SymbolicTransitionSystem,
                           manager: Cudd,
                           verbose: bool = False,
                           plot: bool = False) -> Tuple[List[SymbolicDFA], List[BDD], List[BDD]]:
    """
    A helper function to build a symbolic DFA given a formul from BDD Variables.
    """
    
    # create a list of DFAs
    DFA_handles = []
    DFA_curr_vars = []
    DFA_nxt_vars = []

    for _idx, fmla in enumerate(formulas):
        # create different boolean variables for different DFAs - [a0_i for ith DFA]
        dfa_curr_state, dfa_next_state, _dfa = create_symbolic_dfa_graph(cudd_manager=cudd_manager,
                                                                         formula= fmla,
                                                                         dfa_num=_idx)

        # create TR corresponding to each DFA - dfa name is only used dumping graph 
        dfa_tr = SymbolicDFA(curr_states=dfa_curr_state,
                             next_states=dfa_next_state,
                             predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                             manager=manager,
                             dfa=_dfa,
                             dfa_name=f'dfa_{_idx}')

        dfa_tr.create_dfa_transition_system(verbose=verbose,
                                            plot=plot,
                                            valid_dfa_edge_formula_size=len(_dfa.get_symbols()))

         # We extend DFA vars list as we dont need them in stored in separate lists
        DFA_handles.append(dfa_tr)
        DFA_curr_vars.extend(dfa_curr_state)
        DFA_nxt_vars.extend(dfa_next_state)
    
    return DFA_handles, DFA_curr_vars, DFA_nxt_vars


def build_bdd_abstraction(cudd_manager, problem_file_path, domain_file_path) -> Tuple[SymbolicTransitionSystem, list, list, list, int]:
    """
    Main Function to Build Transition System that only represent valid edges without any weights
    """
    
    if not CREATE_VAR_LBLS:
        task, domain, ts_curr_state, ts_next_state  = create_symbolic_causal_graph(cudd_manager=cudd_manager,
                                                                                problem_pddl_file=problem_file_path,
                                                                                domain_pddl_file=domain_file_path,
                                                                                create_lbl_vars=False,
                                                                                draw_causal_graph=DRAW_EXPLICIT_CAUSAL_GRAPH)
        sym_tr = SymbolicTransitionSystem(curr_states=ts_curr_state,
                                            next_states=ts_next_state,
                                            lbl_states=None,
                                            observations=None,
                                            task=task,
                                            domain=domain,
                                            manager=cudd_manager)

    else:
        task, domain, possible_obs, ts_curr_state, ts_next_state, ts_lbl_states  = create_symbolic_causal_graph(cudd_manager=cudd_manager,
                                                                                                        problem_pddl_file=problem_file_path,
                                                                                                        domain_pddl_file=domain_file_path,
                                                                                                        create_lbl_vars=True,
                                                                                                        max_valid_formula_size=1,
                                                                                                        draw_causal_graph=DRAW_EXPLICIT_CAUSAL_GRAPH)
        sym_tr = SymbolicTransitionSystem(curr_states=ts_curr_state,
                                            next_states=ts_next_state,
                                            lbl_states=ts_lbl_states,
                                            observations=possible_obs,
                                            task=task,
                                            domain=domain,
                                            manager=cudd_manager)


        
        ts_total_state = len(task.facts)
        sym_tr.create_transition_system(verbose=False, plot=False)

        sym_tr.create_state_obs_bdd(verbose=False, plot=False)  

    return  sym_tr, ts_curr_state, ts_next_state, ts_lbl_states, ts_total_state



def build_weighted_add_abstraction(cudd_manager, problem_file_path, domain_file_path, weight_list: list) -> Tuple[SymbolicTransitionSystem, list, list, list, int]:
    """
    Main Function to Build Transition System that represents valid edges without their corresponding weights

    Pyperplan supports the following PDDL fragment: STRIPS without action costs
    """
    # explore a more robust version
    c_a = cudd_manager.addConst(int(1))
    c_b = cudd_manager.addConst(int(1))
    c_c = cudd_manager.addConst(int(1))
    c_d = cudd_manager.addConst(int(1))

    task, domain, possible_obs, add_ts_curr_state, add_ts_next_state, add_ts_lbl_states  = create_symbolic_causal_graph(cudd_manager=cudd_manager,
                                                                                                                    problem_pddl_file=problem_file_path,
                                                                                                                    domain_pddl_file=domain_file_path,
                                                                                                                    create_lbl_vars=True,
                                                                                                                    max_valid_formula_size=1,
                                                                                                                    draw_causal_graph=DRAW_EXPLICIT_CAUSAL_GRAPH,
                                                                                                                    add_flag=True)
    sym_tr = SymbolicWeightedTransitionSystem(curr_states=add_ts_curr_state,
                                              next_states=add_ts_next_state,
                                              lbl_states=add_ts_lbl_states,
                                              observations=possible_obs,
                                              task=task,
                                              domain=domain,
                                              manager=cudd_manager)

    ts_total_state = len(task.facts)
    # All action have equal weight of 1 unit 
    sym_tr.create_weighted_transition_system(verbose=False, plot=False, weight_list=[c_a, c_b, c_c, c_d])

    sym_tr.create_state_obs_add(verbose=False, plot=False)

    return  sym_tr, add_ts_curr_state, add_ts_next_state, add_ts_lbl_states, ts_total_state 

        


if __name__ == "__main__":
    domain_file_path = PROJECT_ROOT + "/pddl_files/grid_world/domain.pddl"
    if OBSTACLE:
        problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}_obstacle1.pddl"
    else:
        problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

    cudd_manager = Cudd()
    
    # Build Transition with costs
    if QUANTITATIVE_SEARCH:
        # All vars (TS, DFA and Predicate) are of type ADDs
        sym_tr, ts_curr_state, ts_next_state, ts_lbl_states, ts_total_state = build_weighted_add_abstraction(cudd_manager=cudd_manager,
                                                                                                             problem_file_path=problem_file_path,
                                                                                                             domain_file_path=domain_file_path,
                                                                                                             weight_list=[1, 1, 1, 1])
        # The tuple contains the DFA handle, DFA curr and next vars in this specific order
        dfa_tr, dfa_curr_state, dfa_next_state = build_add_symbolic_dfa(formulas=formulas,
                                                                        sym_tr_handle=sym_tr,
                                                                        manager=cudd_manager,
                                                                        verbose=False,
                                                                        plot=False)
    # Build Transition with no costs
    else:
        sym_tr, ts_curr_state, ts_next_state, ts_lbl_states, ts_total_state = build_bdd_abstraction(cudd_manager=cudd_manager,
                                                                                                    problem_file_path=problem_file_path,
                                                                                                    domain_file_path=domain_file_path)

        dfa_tr, dfa_curr_state, dfa_next_state = build_bdd_symbolic_dfa(formulas=formulas,
                                                                        sym_tr_handle=sym_tr,
                                                                        manager=cudd_manager,
                                                                        verbose=False,
                                                                        plot=False)

    # sys.exit()
    if DYNAMIC_VAR_ORDERING:
        set_variable_reordering(manager=cudd_manager,
                                make_tree_node=True,
                                ts_sym_var_len=len(ts_curr_state),
                                ts_obs_var_len=len(ts_lbl_states))

    if len(formulas) > 1:
        start: float = time.time()
        if QUANTITATIVE_SEARCH:
            graph_search = MultipleFormulaDijkstra(ts_handle=sym_tr,
                                                   dfa_handle=dfa_tr,
                                                   ts_curr_vars=ts_curr_state,
                                                   ts_next_vars=ts_next_state,
                                                   dfa_curr_vars=dfa_curr_state,
                                                   dfa_next_vars=dfa_next_state,
                                                   ts_obs_vars=ts_lbl_states,
                                                   cudd_manager=cudd_manager)

            # call dijkstras for solving minimum cost path over nLTLs
            action_dict: dict = graph_search.symbolic_dijkstra_nLTL(verbose=False)

        else:
            graph_search = MultipleFormulaBFS(ts_handle=sym_tr,
                                              dfa_handles=dfa_tr,
                                              ts_curr_vars=ts_curr_state,
                                              ts_next_vars=ts_next_state,
                                              dfa_curr_vars=dfa_curr_state,
                                              dfa_next_vars=dfa_next_state,
                                              ts_obs_vars=ts_lbl_states,
                                              cudd_manager=cudd_manager)

            # BFS for multiple will return multiple paths. One of them will be the shortest path. 
            action_dict: dict = graph_search.symbolic_bfs_nLTL(verbose=False)

        stop: float = time.time()
        print("Time took for plannig: ", stop - start)
        if PRINT_STRATEGY:
            for _dfa_state, _ts_dict in action_dict.items():
                print(f"******************Currently in DFA state {_dfa_state}******************")
                for _ts_state, _action in _ts_dict.items(): 
                    print(f"From State {_ts_state} take Action {_action}")

            print("Done with the plan")

    else:
        start: float = time.time()
        if QUANTITATIVE_SEARCH:
            # shortest path graph search with Dijkstras
            graph_search =  SymbolicDijkstraSearch(ts_handle=sym_tr,
                                                   dfa_handle=dfa_tr[0],
                                                   ts_curr_vars=ts_curr_state,
                                                   ts_next_vars=ts_next_state,
                                                   dfa_curr_vars=dfa_curr_state,
                                                   dfa_next_vars=dfa_next_state,
                                                   ts_obs_vars=ts_lbl_states,
                                                   cudd_manager=cudd_manager)
            action_dict = graph_search.composed_symbolic_dijkstra_wLTL(verbose=False)

        else:

            graph_search = SymbolicSearch(ts_handle=sym_tr,
                                          dfa_handle=dfa_tr[0],
                                          manager=cudd_manager,
                                          ts_curr_vars=ts_curr_state,
                                          ts_next_vars=ts_next_state,
                                          ts_obs_vars=ts_lbl_states,
                                          dfa_curr_vars=dfa_curr_state,
                                          dfa_next_vars=dfa_next_state)

            # TODO: In future store startegy as Mealey machine (Finite State Machine)
            # The Mealey machine is a characteristic Function that represents a mapping from
            # current TS state x Obs associated with this state x State of the Automation to Next State in TS and next state in the DFA Automaton
            # TR : S_ts x Obs_bdd x S_dfa x S'_ts x S'_dfa
            action_dict = graph_search.composed_symbolic_bfs_wLTL(verbose=False)


        stop: float = time.time()
        print("Time took for plannig: ", stop - start)
        
    if len(formulas) > 1:
        if SIMULATE_STRATEGY and QUANTITATIVE_SEARCH:
            gridworld_strategy = convert_action_dict_to_gridworld_strategy_nLTL(ts_handle=sym_tr,
                                                                                dfa_handle=dfa_tr,
                                                                                action_map=action_dict,
                                                                                init_state_ts_sym=sym_tr.sym_add_init_states,
                                                                                state_obs_dd=sym_tr.sym_add_state_labels,
                                                                                ts_curr_vars=ts_curr_state,
                                                                                ts_next_vars=ts_next_state,
                                                                                dfa_curr_vars=dfa_curr_state,
                                                                                dfa_next_vars=dfa_next_state,
                                                                                ts_sym_to_curr_map=sym_tr.predicate_add_sym_map_curr.inv)


            create_gridworld(size=GRID_WORLD_SIZE, strategy=gridworld_strategy, init_pos=(0, 0))
        
        
        elif SIMULATE_STRATEGY:
            # plot_policy(action_dict)
            gridworld_strategy = convert_action_dict_to_gridworld_strategy_nLTL(ts_handle=sym_tr,
                                                                                dfa_handle=dfa_tr,
                                                                                action_map=action_dict,
                                                                                init_state_ts_sym=sym_tr.sym_init_states,
                                                                                state_obs_dd=sym_tr.sym_state_labels,
                                                                                ts_curr_vars=ts_curr_state,
                                                                                ts_next_vars=ts_next_state,
                                                                                dfa_curr_vars=dfa_curr_state,
                                                                                dfa_next_vars=dfa_next_state,
                                                                                ts_sym_to_curr_map=sym_tr.predicate_sym_map_curr.inv)

            create_gridworld(size=GRID_WORLD_SIZE, strategy=gridworld_strategy, init_pos=(0, 0))
    else:
        if SIMULATE_STRATEGY and QUANTITATIVE_SEARCH:
            gridworld_strategy = convert_action_dict_to_gridworld_strategy(ts_handle=sym_tr,
                                                                           dfa_handle=dfa_tr[0],
                                                                           action_map=action_dict,
                                                                           init_state_ts=sym_tr.sym_add_init_states,
                                                                           state_obs_dd=sym_tr.sym_add_state_labels,
                                                                           ts_curr_vars=ts_curr_state,
                                                                           ts_next_vars=ts_next_state,
                                                                           dfa_curr_vars=dfa_curr_state,
                                                                           dfa_next_vars=dfa_next_state,
                                                                           ts_sym_to_curr_map=sym_tr.predicate_add_sym_map_curr.inv,
                                                                           dfa_sym_to_curr_map=dfa_tr[0].dfa_predicate_add_sym_map_curr.inv)
            create_gridworld(size=GRID_WORLD_SIZE, strategy=gridworld_strategy, init_pos=(0, 0))

        elif SIMULATE_STRATEGY:
            gridworld_strategy = convert_action_dict_to_gridworld_strategy(ts_handle=sym_tr,
                                                                           dfa_handle=dfa_tr[0],
                                                                           action_map=action_dict,
                                                                           init_state_ts=sym_tr.sym_init_states,
                                                                           state_obs_dd=sym_tr.sym_state_labels,
                                                                           ts_curr_vars=ts_curr_state,
                                                                           ts_next_vars=ts_next_state,
                                                                           dfa_curr_vars=dfa_curr_state,
                                                                           dfa_next_vars=dfa_next_state,
                                                                           ts_sym_to_curr_map=sym_tr.predicate_sym_map_curr.inv,
                                                                           dfa_sym_to_curr_map=dfa_tr[0].dfa_predicate_sym_map_curr.inv)
            create_gridworld(size=GRID_WORLD_SIZE, strategy=gridworld_strategy, init_pos=(0, 0))

    
