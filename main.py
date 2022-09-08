import os
import sys
import copy
import time
import graphviz as gv
import math

from typing import Tuple, List
from functools import reduce
from cudd import Cudd, BDD, ADD
from itertools import product


from src.causal_graph import CausalGraph
from src.two_player_game import TwoPlayerGame
from src.transition_system import FiniteTransitionSystem
from graph_search import graph_search

from src.symbolic_search import SymbolicSearch
from src.two_player_game import TwoPlayerGame
from src.symbolic_dfa import SymbolicDFA, SymbolicAddDFA
from src.symbolic_abstraction import SymbolicTransitionSystem, SymbolicWeightedTransitionSystem
from src.symbolic_quantitative_search import SymbolicDijkstraSearch

import  src.gridworld_visualizer.gridworld_vis.gridworld as gridworld_handle


from config import *

QUANTITATIVE_SEARCH: bool = True  # set this flag to true whe you have edge costs

BUILD_DFA: bool = True
BUILD_ABSTRACTION: bool = True
CREATE_VAR_LBLS: bool = True   # set this to true if you want to create Observation BDDs
# construct a sample rwo player game and wrap it to construct its symbolic version
# transition_graph = get_graph(print_flag=True)
# build_symbolic_model(transition_graph)

DRAW_EXPLICIT_CAUSAL_GRAPH: bool = False
SIMULATE_STRATEGY: bool = False
GRID_WORLD_SIZE: int = 5
OBSTACLE: bool = False  # galf to load the onbstacle gridworl and color the gridworld accordingly


def create_gridworld(size: int, strategy: list, init_pos: tuple = (1, 1)):
    def tile2classes(x, y):
        # draw horizontal block
        if (5 <= x <= 16) and (16 <= y <= 18):
            return "lava"
        
        # draw vertical block
        if (16 <= x <= 18) and (11 <= y <= 18):
            return "lava"

        # elif (x in (0, 7)) and (y in (0, 7)):
        #     return "recharge"
        # elif (2 <= x <= 5) and y in (0, 7):
        #     return "dry"
        # elif x in (1, 6) and (y in (4, 5) or y <= 1):
        #     return "lava"
        # elif (x in (0, 7)) and (y in (1, 4, 5)):
        #     return "lava"

        return "normal"

    file_name = PROJECT_ROOT + f'/plots/simulated_strategy.svg'

    if strategy is None:
        strategy = [gridworld_handle.E, gridworld_handle.N, gridworld_handle.N,
                    gridworld_handle.N, gridworld_handle.N, gridworld_handle.W, 
                    gridworld_handle.W, gridworld_handle.W]

    svg = gridworld_handle.gridworld(n=size, tile2classes=tile2classes, actions=strategy, init_pos=init_pos)
    svg.saveas(file_name, pretty=True)


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
    # create all combinations of locations - for gridworld you repeat -1
    # possible_obs = list(product(lbls, repeat=1))

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

    # we subtract -2 as we have (skbn) and (!(skbn))(an agent) defined as an object. We do not need to allocate a var for this agent. 

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
    _num_of_sym_vars = cudd_manager.size()

    for num_var in range(math.ceil(math.log2(num_of_facts))):
            if add_flag:
                curr_state_vars.append(manager.addVar(_num_of_sym_vars + (2*num_var), f'{cur_state}{num_var}'))
                next_state_vars.append(manager.addVar(_num_of_sym_vars + (2*num_var + 1), f'{nxt_state}{num_var}'))
            else:
                curr_state_vars.append(manager.bddVar(_num_of_sym_vars + (2*num_var), f'{cur_state}{num_var}'))
                next_state_vars.append(manager.bddVar(_num_of_sym_vars + (2*num_var + 1), f'{nxt_state}{num_var}'))

    return (curr_state_vars, next_state_vars)


def create_symbolic_dfa_graph(cudd_manager, formula: str, dfa_num: int, add_flag: bool = False):
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


def get_graph(print_flag: bool = False):

    _domain_file_path = PROJECT_ROOT + + "/pddl_files/two_table_scenario/arch/domain.pddl"
    _problem_file_path = PROJECT_ROOT + + "/pddl_files/two_table_scenario/arch/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=False)

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


def convert_action_dict_to_gridworld_strategy(action_map: dict,
                                              transition_sys_tr,
                                              tr_action_idx_map,
                                              dfa_tr,
                                              init_state_ts,
                                              init_state_dfa,
                                              target_DFA,
                                              state_obs_bdd,
                                              ts_curr_vars: list,
                                              ts_next_vars: list,
                                              dfa_curr_vars: list,
                                              dfa_next_vars: list,
                                              ts_sym_to_curr_map,
                                              dfa_sym_to_curr_map) -> List:
    """
    A helper function that compute a sequence of Gridworld based sequence of actions from the strategy compute by Graph search algorithm.

    Approach: 
    1. We start from the initial state in the TS and the DFA, we get the strategy from the current DFA and TS state, 
    2. We then convert it Gridworld friendly cardinal action (N, S, E, W)
    3. Repeat until we reach an acceepting state.
    """
    _map = {'moveright': gridworld_handle.E,
            'moveleft': gridworld_handle.W,
            'moveup': gridworld_handle.N,
            'movedown': gridworld_handle.S }
    
    _strategy = []

    curr_ts_state = init_state_ts
    curr_dfa_state = init_state_dfa

    while not target_DFA == curr_dfa_state:
        # get the strategy
        try:
            _a = action_map[dfa_sym_to_curr_map[curr_dfa_state]][ts_sym_to_curr_map[curr_ts_state]]
        except:
            return _strategy

        try:
            _strategy.append(_map.get(_a))
        except:
            _strategy.append(_map.get(_a[0]))

        # get the next TS
        try:
            _nxt_ts_state = transition_sys_tr[tr_action_idx_map[_a]].restrict(curr_ts_state)
        except:
            _nxt_ts_state = transition_sys_tr[tr_action_idx_map[_a[0]]].restrict(curr_ts_state)
        
        _nxt_ts_state = _nxt_ts_state.swapVariables(ts_curr_vars, ts_next_vars)
        
        # get the observation of this ts state
        _sym_obs = state_obs_bdd.restrict(_nxt_ts_state)

        # get the next DFA state
        _nxt_dfa = dfa_tr.restrict(curr_dfa_state & _sym_obs)
        
        # finally swap variables of TS and DFA 
        curr_ts_state = _nxt_ts_state
        curr_dfa_state = _nxt_dfa.swapVariables(dfa_curr_vars, dfa_next_vars)

    return _strategy


def build_add_symbolic_dfa(formulas: List[str],
                           add_ts_lbl_states: List[ADD],
                           sym_tr_handle: SymbolicWeightedTransitionSystem,
                           manager: Cudd) -> TwoPlayerGame:
    """
    A helper function to build a symbolic DFA given a formula from ADD Variables.
    """
    
    # create a list of DFAs
    DFA_list = []

    # for now dfa_num is zero. If oyu rhave multiple DFAs then loop over them and update DFA_num
    for _idx, fmla in enumerate(formulas):
        # create boolean ADD variables
        add_dfa_curr_state, add_dfa_next_state, _dfa = create_symbolic_dfa_graph(cudd_manager=cudd_manager,
                                                                                 formula= fmla,
                                                                                 dfa_num=_idx,
                                                                                 add_flag=True)
        # create TR corresponding to each DFA - dfa name is only used dumping graph 
        dfa_tr = SymbolicAddDFA(curr_states=add_dfa_curr_state,
                                next_states=add_dfa_next_state,
                                ts_lbls=add_ts_lbl_states,
                                predicate_add_sym_map_lbl=sym_tr_handle.predicate_add_sym_map_lbl,
                                predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                manager=manager,
                                dfa=_dfa,
                                dfa_name=f'dfa_{_idx}')

        dfa_tr.create_dfa_transition_system(verbose=True,
                                            plot=True,
                                            valid_dfa_edge_formula_size=len(_dfa.get_symbols()))
    
    return dfa_tr, add_dfa_curr_state, add_dfa_next_state



def build_bdd_symbolic_dfa(formulas: List[str], ts_lbl_states: List[BDD], sym_tr_handle: SymbolicTransitionSystem, manager: Cudd) -> TwoPlayerGame:
    """
    A helper function to build a symbolic DFA given a formul from BDD Variables.
    """
    
    # create a list of DFAs
    DFA_list = []

    # for now dfa_num is zero. If oyu rhave multiple DFAs then loop over them and update DFA_num
    for _idx, fmla in enumerate(formulas):
        # create boolean variables
        dfa_curr_state, dfa_next_state, _dfa = create_symbolic_dfa_graph(cudd_manager=cudd_manager,
                                                                         formula= fmla,
                                                                         dfa_num=_idx)
        # create TR corresponding to each DFA - dfa name is only used dumping graph 
        dfa_tr = SymbolicDFA(curr_states=dfa_curr_state,
                                next_states=dfa_next_state,
                                ts_lbls=ts_lbl_states,
                                predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                manager=manager,
                                dfa=_dfa,
                                dfa_name=f'dfa_{_idx}')

        dfa_tr.create_dfa_transition_system(verbose=False,
                                            plot=False,
                                            valid_dfa_edge_formula_size=len(_dfa.get_symbols()))
    
    return dfa_tr, dfa_curr_state, dfa_next_state


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
    # c_b = cudd_manager.addConst(int(2))
    # c_c = cudd_manager.addConst(int(3))
    # c_d = cudd_manager.addConst(int(4))

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
    # All action are have equal weight of 1 unit 
    sym_tr.create_weighted_transition_system(verbose=True, plot=True, weight_list=[c_a, c_a, c_a, c_a])

    sym_tr.create_state_obs_add(verbose=True, plot=True)

    return  sym_tr, add_ts_curr_state, add_ts_next_state, add_ts_lbl_states, ts_total_state 

        


if __name__ == "__main__":
    domain_file_path = PROJECT_ROOT + "/pddl_files/grid_world/domain.pddl"
    if OBSTACLE:
        problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}_obstacle1.pddl"
    else:
        problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

    cudd_manager = Cudd()

    # list of formula
    formulas = [
        'F(l13)',
        # 'F(l7 & F(l13))',   # simple Formula w 2 states
        # 'F(l13 & (F(l21) & F(l5)))',
        # 'F(l6) & F(l2)', 
        # 'F(l13 & (F(l21 & (F(l5)))))',
        # "F(l21 & (F(l5 & (F(l25 & F(l1))))))",   # traversing the gridworld on the corners
        # "F(l91 & (F(l10 & (F(l100 & F(l1))))))"   # traversing the gridworld on the corners for 10 x 10 gridworld
        # "F(l400)", 
        # "F(l381 & (F(l20 & (F(l400 & F(l1))))))"   # traversing the gridworld on the corners for 20 x 20 gridworld
        ]
    
    # Build Transition with costs
    if QUANTITATIVE_SEARCH:
        # As our Characteristic function associated with the set of valid transitions has costs associated with it, we build this TR using ADD Variables.
        # As Each state has an associated label with (ADD State2OBs), we need to create Label Vars out ADD VAriables too. 
        # As each each edge in DFA is dictated by a symbol (State label), the DFA states are also built out of ADD Variables. 
        sym_tr, ts_curr_state, ts_next_state, ts_lbl_states, ts_total_state = build_weighted_add_abstraction(cudd_manager=cudd_manager,
                                                                                                            problem_file_path=problem_file_path,
                                                                                                            domain_file_path=domain_file_path,
                                                                                                            weight_list=[1, 1, 1, 1])
        dfa_tr, dfa_curr_state, dfa_next_state = build_add_symbolic_dfa(formulas=formulas,
                                                                        add_ts_lbl_states=ts_next_state,
                                                                        sym_tr_handle=sym_tr,
                                                                        manager=cudd_manager)
        
    else:
        # Build Transition with no costs
        sym_tr, ts_curr_state, ts_next_state, ts_lbl_states, ts_total_state = build_bdd_abstraction(cudd_manager=cudd_manager,
                                                                                                    problem_file_path=problem_file_path,
                                                                                                    domain_file_path=domain_file_path)
        
        dfa_tr, dfa_curr_state, dfa_next_state = build_bdd_symbolic_dfa(formulas=formulas,
                                                                        ts_lbl_states=ts_next_state,
                                                                        sym_tr_handle=sym_tr,
                                                                        manager=cudd_manager)
    
    sys.exit()
    # init_state = sym_tr.sym_init_states & dfa_tr.sym_init_state
    # print('Inital states: ', init_state)
    # print('Goal states: ', dfa_tr.sym_goal_state)
    
    giant_tr = reduce(lambda a, b: a | b, sym_tr.sym_tr_actions) 

    # let do graph search from init state to a goal state
    start: float = time.time()
    if QUANTITATIVE_SEARCH:
         # shortest path graph search with Dijkstras
        graph_search =  SymbolicDijkstraSearch(init_TS=sym_tr.sym_add_init_states,
                                            target_DFA=dfa_tr.sym_goal_state,
                                            init_DFA=dfa_tr.sym_init_state,
                                            ts_curr_vars=ts_curr_state,
                                            ts_next_vars=ts_next_state,
                                            dfa_curr_vars=ts_lbl_states,
                                            dfa_next_vars=dfa_curr_state,
                                            ts_obs_vars=ts_lbl_states,
                                            ts_transition_func=giant_tr,
                                            ts_trans_func_list=sym_tr.sym_tr_actions,
                                            dfa_transition_func=dfa_tr.dfa_bdd_tr,
                                            ts_add_sym_to_curr_map=sym_tr.predicate_add_sym_map_curr.inv,
                                            ts_bdd_sym_to_curr_map=sym_tr.predicate_sym_map_curr.inv,
                                            ts_sym_to_S2O_map=sym_tr.predicate_sym_map_lbl.inv,
                                            dfa_sym_to_curr_map=dfa_tr.dfa_predicate_sym_map_curr.inv,
                                            tr_action_idx_map=sym_tr.tr_action_idx_map, 
                                            state_obs_bdd=sym_tr.sym_state_labels,
                                            cudd_manager=cudd_manager)
        
        graph_search.symbolic_dijkstra(verbose=True)
    else:

        graph_search = SymbolicSearch(init=sym_tr.sym_init_states,
                                    target=sym_tr.sym_goal_states,
                                    init_TS=sym_tr.sym_init_states,
                                    target_DFA=dfa_tr.sym_goal_state,
                                    init_DFA=dfa_tr.sym_init_state,
                                    manager=cudd_manager,
                                    ts_curr_vars=ts_curr_state,
                                    ts_next_vars=ts_next_state,
                                    ts_obs_vars=ts_lbl_states,
                                    dfa_curr_vars=dfa_curr_state,
                                    dfa_next_vars=dfa_next_state,
                                    ts_trans_func_list=sym_tr.sym_tr_actions,
                                    ts_transition_func=giant_tr,
                                    dfa_transition_func=dfa_tr.dfa_bdd_tr,
                                    ts_sym_to_curr_map=sym_tr.predicate_sym_map_curr.inv,
                                    ts_sym_to_S2O_map=sym_tr.predicate_sym_map_lbl.inv,
                                    dfa_sym_to_curr_map=dfa_tr.dfa_predicate_sym_map_curr.inv,
                                    tr_action_idx_map=sym_tr.tr_action_idx_map,
                                    state_obs_bdd=sym_tr.sym_state_labels)

        # TODO: In future store startegy as Mealey machine (Finite State Machine)
        # The Mealey machine is a characteristic Function that represents a mapping from
        # current TS state x Obs associated with this state x State of the Automation to Next State in TS and next state in the DFA Automaton
        # TR : S_ts x Obs_bdd x S_dfa x S'_ts x S'_dfa
        action_dict = graph_search.symbolic_bfs_wLTL(max_ts_state=ts_total_state, verbose=False)


    stop: float = time.time()
    print("Time took for plannig: ", stop - start)
    print("Sequence of actions")
    
    for _dfa_state, _ts_dict in action_dict.items():
        print(f"******************Currently in DFA state {_dfa_state}******************")
        for _ts_state, _action in _ts_dict.items(): 
            print(f"From State {_ts_state} take Action {_action}")

    print("Done with the plan")

    if SIMULATE_STRATEGY:
        gridworld_strategy = convert_action_dict_to_gridworld_strategy(action_map=action_dict,
                                                                       transition_sys_tr=sym_tr.sym_tr_actions,
                                                                       dfa_tr=dfa_tr.dfa_bdd_tr,
                                                                       init_state_ts=sym_tr.sym_init_states,
                                                                       init_state_dfa=dfa_tr.sym_init_state,
                                                                       target_DFA=dfa_tr.sym_goal_state,
                                                                       tr_action_idx_map=sym_tr.tr_action_idx_map,
                                                                       state_obs_bdd=sym_tr.sym_state_labels,
                                                                       ts_curr_vars=ts_curr_state,
                                                                       ts_next_vars=ts_next_state,
                                                                       dfa_curr_vars=dfa_curr_state,
                                                                       dfa_next_vars=dfa_next_state,
                                                                       ts_sym_to_curr_map=sym_tr.predicate_sym_map_curr.inv,
                                                                       dfa_sym_to_curr_map=dfa_tr.dfa_predicate_sym_map_curr.inv)
        create_gridworld(size=GRID_WORLD_SIZE, strategy=gridworld_strategy, init_pos=(0, 0))

    sys.exit()
    
    print("Time took for plannig: ", stop - start)


    # graph_search(init=sym_tr.sym_init_states,
    #              target=sym_tr.sym_goal_states,
    #              transition_func=giant_tr,
    #              x_list=curr_state,
    #              y_list=next_state)


    
