import os
import re
import warnings
import random

import src.gridworld_visualizer.gridworld_vis.gridworld as gridworld_handle
import src.gridworld_visualizer.gridworld_vis.matplotlib_gw as policy_plotter

from typing import List, Union

from cudd import Cudd, BDD, ADD
from functools import reduce

from config import PROJECT_ROOT, GRID_WORLD_SIZE, OBSTACLE

from src.algorithms.base.base_symbolic_search import BaseSymbolicSearch
from src.symbolic_graphs import SymbolicDFA, SymbolicAddDFA, SymbolicDFAFranka
from src.symbolic_graphs import SymbolicTransitionSystem, SymbolicWeightedTransitionSystem, SymbolicFrankaTransitionSystem



PDDL_TO_GRIDWORLD_MAP = {'moveright': gridworld_handle.E,
                         'moveleft': gridworld_handle.W,
                         'moveup': gridworld_handle.N,
                         'movedown': gridworld_handle.S}


def plot_policy(action_dict):
    """
    A function to plot the gridworld policy using my policy plotting code.
    """
    

    plot_handle = policy_plotter.plotterClass(fig_title='gridworld_str')
    plot_handle.plot_policy(width=GRID_WORLD_SIZE, height=GRID_WORLD_SIZE, sys_str=action_dict)
    
    file_name = PROJECT_ROOT + f'/plots/{plot_handle.fig_title}_N_{GRID_WORLD_SIZE}.png'
    plot_handle.save_fig(file_name, dpi=500)
    # plot_handle.close()


def get_ADD_dfa_evolution(dfa_handle, _nxt_ts_state, state_obs_dd, dfa_curr_vars, dfa_next_vars, curr_dfa_state_tuple) -> tuple:
    """
    A function that compute the next states on each DFA given a state or a set of states

    add_func: The add associated with the set of states of the Trasition System
    from_dfa_states: The states of DFA states where you are right now. 
    """
    dfa_list = [0 for _ in range(len(dfa_handle.dfa_add_tr_list))]

    curr_dfa_bdd_list = map_dfa_tuple_to_sym_states(dfa_tuple=curr_dfa_state_tuple,
                                                    dfa_sym_to_curr_state_map=dfa_handle.dfa_predicate_add_sym_map_curr.inv,
                                                    dfa_state_int_map=dfa_handle.node_int_map_dfas)
        
    for dfa_idx, dfa_tr in enumerate(dfa_handle.dfa_add_tr_list):
        # check where you evolve in each DFA
        state_obs = state_obs_dd.restrict(_nxt_ts_state)
        dfa_state: ADD = dfa_tr.restrict(state_obs & curr_dfa_bdd_list[dfa_idx])
        dfa_state: ADD = dfa_state.swapVariables(dfa_curr_vars, dfa_next_vars)
        dfa_state: str = re.split('_\d', dfa_handle.dfa_predicate_add_sym_map_curr.inv[dfa_state])[0]
        dfa_list[dfa_idx] = dfa_handle.node_int_map_dfas[dfa_idx][dfa_state]
    
    curr_dfa_state_tuple = tuple(dfa_list)
        
    return curr_dfa_state_tuple



def get_dfa_evolution(dfa_handle, _nxt_ts_state, state_obs_dd, dfa_curr_vars, dfa_next_vars, curr_dfa_state_tuple):
    """
    A function to compute the next state of the DFA when evolving over the product of multiple tasks
    """
    
    # get the observation of this ts state
    dfa_list = [0 for _ in range(len(dfa_handle.dfa_bdd_tr_list))]
    curr_dfa_bdd_list = map_dfa_tuple_to_sym_states(dfa_tuple=curr_dfa_state_tuple,
                                                    dfa_sym_to_curr_state_map=dfa_handle.dfa_predicate_sym_map_curr.inv,
                                                    dfa_state_int_map=dfa_handle.node_int_map_dfas)
    # get the next dfa tuple 
    for dfa_idx, dfa_tr in enumerate( dfa_handle.dfa_bdd_tr_list):
        # check where you evolve in each DFA
        state_obs = state_obs_dd.restrict(_nxt_ts_state)
        dfa_state = dfa_tr.restrict(state_obs & curr_dfa_bdd_list[dfa_idx])
        dfa_state = dfa_state.swapVariables(dfa_curr_vars, dfa_next_vars)
        dfa_state = re.split('_\d', dfa_handle.dfa_predicate_sym_map_curr.inv[dfa_state])[0]
        dfa_list[dfa_idx] = dfa_handle.node_int_map_dfas[dfa_idx][dfa_state]

    curr_dfa_state_tuple = tuple(dfa_list)

    return curr_dfa_state_tuple


def create_gridworld(size: int, strategy: list, file_name: str = ' ', init_pos: tuple = (1, 1)):
    def tile2classes_obstacle(x, y):
        # draw horizontal block
        if (5 <= x <= 16) and (16 <= y <= 18):
            return "lava"
        
        # draw vertical block
        if (16 <= x <= 18) and (11 <= y <= 18):
            return "lava"

        return "normal"
    
    def tile2classes(x, y):

        return "normal"

    if file_name == ' ':
        file_name = PROJECT_ROOT + '/plots/simulated_strategy.svg'
    else:
        assert 'svg' in file_name, "Please make sure the save strategy has .svg as file extension."
        file_name = PROJECT_ROOT + f'/plots/{file_name}'

    if strategy is None:
        warnings.warn("Strategy is None. Unrolling Default strategy")
        strategy = [gridworld_handle.E, gridworld_handle.N, gridworld_handle.N,
                    gridworld_handle.N, gridworld_handle.N, gridworld_handle.W, 
                    gridworld_handle.W, gridworld_handle.W]
    if OBSTACLE:
        svg = gridworld_handle.gridworld(n=size, tile2classes=tile2classes_obstacle, actions=strategy, init_pos=init_pos)
    else:
        svg = gridworld_handle.gridworld(n=size, tile2classes=tile2classes, actions=strategy, init_pos=init_pos)
    
    assert os.path.isdir(PROJECT_ROOT + f'/plots') is True, "Please ensure that you have a folder `plots` in the project root to save simulated gridwork policy."

    svg.saveas(file_name, pretty=True)


# helper methods copied as is from nLTL BFS file
def map_dfa_state_to_tuple(dfa_states: List[BDD], dfa_sym_to_curr_state_map: dict, dfa_state_int_map: dict) -> tuple:
    """
    Given a list of sybolic DFA state, create  
    """
    _to_tuple = []
    for dfa_idx, _sym_s in enumerate(dfa_states):
        _s = dfa_sym_to_curr_state_map[_sym_s]
        dfa_state = re.split('_\d', _s)[0]
        _to_tuple.append(dfa_state_int_map[dfa_idx][dfa_state])
    
    return tuple(_to_tuple)


def map_dfa_tuple_to_sym_states(dfa_tuple: tuple, dfa_sym_to_curr_state_map: dict, dfa_state_int_map: dict) -> List[str]:
    """
    Given a tuple, this function returns a list of DFA states
    """
    _dfa_states = []
    for dfa_idx, _state_num in enumerate(dfa_tuple):
        _state_name = dfa_state_int_map[dfa_idx].inv[_state_num]
        _dfa_states.append(dfa_sym_to_curr_state_map.inv[f'{_state_name}_{dfa_idx}'])
    return _dfa_states



def convert_action_dict_to_gridworld_strategy_nLTL(ts_handle: Union[SymbolicWeightedTransitionSystem, SymbolicTransitionSystem],
                                                   dfa_handles: Union[List[SymbolicAddDFA], List[ SymbolicDFA]],
                                                   action_map: dict,
                                                   init_state_ts_sym,
                                                   state_obs_dd,
                                                   ts_curr_vars: list,
                                                   ts_next_vars: list,
                                                   dfa_curr_vars: list,
                                                   dfa_next_vars: list) -> List:
    _strategy = []
    ADD_flag: bool = False

    transition_sys_tr = ts_handle.sym_tr_actions
    tr_action_idx_map = ts_handle.tr_action_idx_map

    dfa_transition_fun_list = [dfa_tr.dfa_bdd_tr for dfa_tr in dfa_handles]
    dfa_monolithic_tr_func = reduce(lambda a, b: a & b,  dfa_transition_fun_list)

    init_state_dfa_list = [dfa_tr.sym_init_state for dfa_tr in dfa_handles]
    target_DFA_list = [dfa_tr.sym_goal_state for dfa_tr in dfa_handles]

    curr_ts_state_sym = init_state_ts_sym

    monolithic_dfa_init = reduce(lambda x, y: x & y, init_state_dfa_list)
    monolithic_dfa_target = reduce(lambda x, y: x & y, target_DFA_list)

    curr_dfa_state = monolithic_dfa_init


    if isinstance(init_state_dfa_list[0], ADD):
        ADD_flag = True
   
    counter = 0
    while not curr_dfa_state == monolithic_dfa_target:    
        if ADD_flag:
            _a = action_map[curr_dfa_state.bddPattern() & curr_ts_state_sym.bddPattern()]
        else:
            _a = action_map[curr_dfa_state & curr_ts_state_sym]
        if isinstance(_a, list):
            # randomly select an action from a list of actions
            _a = random.choice(_a)
            _strategy.append(PDDL_TO_GRIDWORLD_MAP.get(_a))
        else:
            _strategy.append(PDDL_TO_GRIDWORLD_MAP.get(_a))
        
        _nxt_ts_state_sym = transition_sys_tr[tr_action_idx_map[_a]].restrict(curr_ts_state_sym)
        _nxt_ts_state_sym = _nxt_ts_state_sym.swapVariables(ts_curr_vars, ts_next_vars)
        
        if ADD_flag:
            # remove the dependency on the weight by first converting it to BDD and then back to ADD.
            _nxt_ts_state_sym = _nxt_ts_state_sym.bddPattern().toADD()

        # get the observation of this ts state
        _sym_obs = state_obs_dd.restrict(_nxt_ts_state_sym)

        # get the next DFA state
        _nxt_dfa = dfa_monolithic_tr_func.restrict(curr_dfa_state & _sym_obs)

        # finally swap variables of TS and DFA 
        curr_ts_state_sym = _nxt_ts_state_sym
        curr_dfa_state = _nxt_dfa.swapVariables(dfa_curr_vars, dfa_next_vars)

        assert counter <= len(action_map), "Error while extracting a valid plan from multiple DFAs."
        counter += 1

    return _strategy

        
def convert_action_dict_to_gridworld_strategy(ts_handle: Union[SymbolicWeightedTransitionSystem, SymbolicTransitionSystem],
                                              dfa_handle: Union[SymbolicAddDFA, SymbolicDFA],
                                              action_map: dict,
                                              init_state_ts,
                                              state_obs_dd,
                                              ts_curr_vars: list,
                                              ts_next_vars: list,
                                              dfa_curr_vars: list,
                                              dfa_next_vars: list) -> List:
    """
    A helper function that compute a sequence of Gridworld based sequence of actions from the strategy compute by Graph search algorithm.

    Approach: 
    1. We start from the initial state in the TS and the DFA, we get the strategy from the current DFA and TS state, 
    2. We then convert it Gridworld friendly cardinal action (N, S, E, W)
    3. Repeat until we reach an acceepting state.
    """
    
    _strategy = []
    ADD_flag: bool = False
    transition_sys_tr = ts_handle.sym_tr_actions
    tr_action_idx_map = ts_handle.tr_action_idx_map

    dfa_tr = dfa_handle.dfa_bdd_tr
    init_state_dfa = dfa_handle.sym_init_state
    target_DFA = dfa_handle.sym_goal_state

    
    if isinstance(init_state_dfa, ADD):
        ADD_flag = True
    curr_ts_state = init_state_ts
    curr_dfa_state = init_state_dfa

    while not target_DFA == curr_dfa_state:
        # get the strategy
        if ADD_flag:
            _a = action_map[curr_dfa_state.bddPattern() & curr_ts_state.bddPattern()]
        else:
            _a = action_map[curr_dfa_state & curr_ts_state]

        if isinstance(_a, list):
            # randomly select an action from a list of actions
            _a = random.choice(_a)
            _strategy.append(PDDL_TO_GRIDWORLD_MAP.get(_a))
        else:
            _strategy.append(PDDL_TO_GRIDWORLD_MAP.get(_a))
        
        # get the next TS
        _nxt_ts_state = transition_sys_tr[tr_action_idx_map[_a]].restrict(curr_ts_state)
        _nxt_ts_state = _nxt_ts_state.swapVariables(ts_curr_vars, ts_next_vars)
        if ADD_flag:
            # remove the dependency on the weight by first converting it to BDD and then back to ADD.
            _nxt_ts_state = _nxt_ts_state.bddPattern().toADD()
        
        # get the observation of this ts state
        _sym_obs = state_obs_dd.restrict(_nxt_ts_state)

        # get the next DFA state
        _nxt_dfa = dfa_tr.restrict(curr_dfa_state & _sym_obs)
        
        # finally swap variables of TS and DFA 
        curr_ts_state = _nxt_ts_state
        curr_dfa_state = _nxt_dfa.swapVariables(dfa_curr_vars, dfa_next_vars)

    return _strategy


def roll_out_franka_strategy_nLTL(ts_handle: Union[SymbolicWeightedTransitionSystem, SymbolicTransitionSystem],
                                  dfa_handles: Union[List[SymbolicAddDFA], List[ SymbolicDFA]],
                                  action_map: dict,
                                  init_state_ts_sym,
                                  state_obs_dd,
                                  ts_curr_vars: list,
                                  ts_next_vars: list,
                                  dfa_curr_vars: list,
                                  dfa_next_vars: list) -> List[str]:
    
    _strategy = []
    ADD_flag: bool = False

    transition_sys_tr = ts_handle.sym_tr_actions
    tr_action_idx_map = ts_handle.tr_action_idx_map

    dfa_transition_fun_list = [dfa_tr.dfa_bdd_tr for dfa_tr in dfa_handles]
    dfa_monolithic_tr_func = reduce(lambda a, b: a & b,  dfa_transition_fun_list)

    init_state_dfa_list = [dfa_tr.sym_init_state for dfa_tr in dfa_handles]
    target_DFA_list = [dfa_tr.sym_goal_state for dfa_tr in dfa_handles]

    curr_ts_state_sym = init_state_ts_sym

    monolithic_dfa_init = reduce(lambda x, y: x & y, init_state_dfa_list)
    monolithic_dfa_target = reduce(lambda x, y: x & y, target_DFA_list)

    curr_dfa_state = monolithic_dfa_init

    if isinstance(init_state_dfa_list[0], ADD):
        ADD_flag = True
        ts_bdd_curr_vars = [_avar.bddPattern() for _avar in ts_curr_vars]
        dfa_bdd_curr_vars = [var.bddPattern() for dfa_tr in dfa_handles for var in dfa_tr.sym_add_vars_curr]
    else:
        # get all th dfa vars 
        monolithic_dfa_vars = [var for dfa_tr in dfa_handles for var in dfa_tr.sym_vars_curr]
        monolithic_dfa_xcube = reduce(lambda x, y: x & y, monolithic_dfa_vars)

    counter = 0

    while not curr_dfa_state == monolithic_dfa_target:    
        # get the strategy
        if ADD_flag:
            indv_state = ts_handle._convert_state_lbl_cube_to_func(dd_func=(curr_ts_state_sym & curr_dfa_state).bddPattern(),
                                                                   prod_curr_list=ts_bdd_curr_vars + dfa_bdd_curr_vars)
        else:
            indv_state = ts_handle._convert_state_lbl_cube_to_func(dd_func=curr_ts_state_sym & curr_dfa_state,
                                                                   prod_curr_list=ts_curr_vars + dfa_curr_vars)
        for prod_state in indv_state:
            if prod_state in action_map:
                
                if ADD_flag:
                    curr_ts_state = prod_state.existAbstract(reduce(lambda x, y: x & y, dfa_bdd_curr_vars))
                    curr_dfa_state = prod_state.existAbstract(reduce(lambda x, y: x & y, ts_bdd_curr_vars))

                    _a = action_map[prod_state]

                    ts_state_tuple = ts_handle.predicate_sym_map_curr.inv[curr_ts_state]
                    ts_state_name = [ts_handle.pred_int_map.inv[pred] for pred in ts_state_tuple]
                    curr_ts_state = curr_ts_state.toADD()
                    curr_dfa_state = curr_dfa_state.toADD()
                else:
                    curr_ts_state_sym = prod_state.existAbstract(monolithic_dfa_xcube)
                    curr_dfa_state = prod_state.existAbstract(reduce(lambda x, y: x & y, ts_handle.sym_vars_curr))
                    _a = action_map[prod_state]
                    ts_state_tuple = ts_handle.predicate_sym_map_curr.inv[curr_ts_state_sym]
                    ts_state_name = [ts_handle.pred_int_map.inv[pred] for pred in ts_state_tuple]
                break
        
        if isinstance(_a, list):
            # randomly select an action from a list of actions
            _a = random.choice(_a)
            _strategy.append((ts_state_name ,_a))
        else:
            _strategy.append((ts_state_name ,_a))
        
        _nxt_ts_state_sym = transition_sys_tr[tr_action_idx_map[_a]].restrict(curr_ts_state_sym)
        _nxt_ts_state_sym = _nxt_ts_state_sym.swapVariables(ts_curr_vars, ts_next_vars)
        
        if ADD_flag:
            # remove the dependency on the weight by first converting it to BDD and then back to ADD.
            _nxt_ts_state_sym = _nxt_ts_state_sym.bddPattern().toADD()

        # get the observation of this ts state
        _sym_obs = state_obs_dd.restrict(_nxt_ts_state_sym)

        # get the next DFA state
        _nxt_dfa = dfa_monolithic_tr_func.restrict(curr_dfa_state & _sym_obs)

        if ADD_flag:
            _nxt_dfa = _nxt_dfa.existAbstract(reduce(lambda x, y: x & y, ts_handle.sym_add_vars_lbl))
            _nxt_dfa = _nxt_dfa.bddPattern().toADD()
        else:
            _nxt_dfa = _nxt_dfa.existAbstract(reduce(lambda x, y: x & y, ts_handle.sym_vars_lbl))

        # finally swap variables of TS and DFA 
        curr_ts_state_sym = _nxt_ts_state_sym
        curr_dfa_state = _nxt_dfa.swapVariables(dfa_curr_vars, dfa_next_vars)

        assert counter <= len(action_map), "Error while extracting a valid plan from multiple DFAs."
        counter += 1

    return _strategy



def roll_out_franka_strategy(ts_handle: Union[SymbolicWeightedTransitionSystem, SymbolicFrankaTransitionSystem],
                             dfa_handle: Union[SymbolicAddDFA, SymbolicDFA],
                             action_map: dict,
                             init_state_ts,
                             state_obs_dd,
                             ts_curr_vars: list,
                             ts_next_vars: list,
                             dfa_curr_vars: list,
                             dfa_next_vars: list) -> List[str]:
    
    _strategy = []
    ADD_flag: bool = False
    transition_sys_tr = ts_handle.sym_tr_actions
    tr_action_idx_map = ts_handle.tr_action_idx_map

    dfa_tr = dfa_handle.dfa_bdd_tr
    init_state_dfa = dfa_handle.sym_init_state
    target_DFA = dfa_handle.sym_goal_state

    
    if isinstance(init_state_dfa, ADD):
        ADD_flag = True
        ts_bdd_curr_vars = [_avar.bddPattern() for _avar in ts_curr_vars]
        dfa_bdd_curr_vars = [_avar.bddPattern() for _avar in dfa_curr_vars]

    curr_ts_state = init_state_ts
    curr_dfa_state = init_state_dfa

    counter = 0


    while not target_DFA == curr_dfa_state:
        # get the strategy
        if ADD_flag:
            indv_state = ts_handle._convert_state_lbl_cube_to_func(dd_func=(curr_ts_state & curr_dfa_state).bddPattern(),
                                                                   prod_curr_list=ts_bdd_curr_vars + dfa_bdd_curr_vars)
        else:
            indv_state = ts_handle._convert_state_lbl_cube_to_func(dd_func=curr_ts_state & curr_dfa_state,
                                                                prod_curr_list=ts_curr_vars + dfa_curr_vars)
        for prod_state in indv_state:
            if prod_state in action_map:
                if ADD_flag:
                    curr_ts_state = prod_state.existAbstract(reduce(lambda x, y: x & y, dfa_bdd_curr_vars))
                    curr_dfa_state = prod_state.existAbstract(reduce(lambda x, y: x & y, ts_bdd_curr_vars))
                    _a = action_map[prod_state]
                    ts_state_tuple = ts_handle.predicate_sym_map_curr.inv[curr_ts_state]
                    ts_state_name = [ts_handle.pred_int_map.inv[pred] for pred in ts_state_tuple]
                    curr_ts_state = curr_ts_state.toADD()
                    curr_dfa_state = curr_dfa_state.toADD()
                else:
                    curr_ts_state = prod_state.existAbstract(reduce(lambda x, y: x & y, dfa_handle.sym_vars_curr))
                    curr_dfa_state = prod_state.existAbstract(reduce(lambda x, y: x & y, ts_handle.sym_vars_curr))
                    _a = action_map[prod_state]
                    ts_state_tuple = ts_handle.predicate_sym_map_curr.inv[curr_ts_state]
                    ts_state_name = [ts_handle.pred_int_map.inv[pred] for pred in ts_state_tuple]
                break

        if isinstance(_a, list):
            # randomly select an action from a list of actions
            _a = random.choice(_a)
            _strategy.append((ts_state_name ,_a))
        else:
            _strategy.append((ts_state_name, _a))
        
        # get the next TS
        _nxt_ts_state = transition_sys_tr[tr_action_idx_map[_a]].restrict(curr_ts_state)
        _nxt_ts_state = _nxt_ts_state.swapVariables(ts_curr_vars, ts_next_vars)
        
        if ADD_flag:
            # remove the dependency on the weight by first converting it to BDD and then back to ADD.
            _nxt_ts_state = _nxt_ts_state.bddPattern().toADD()
        
        # get the observation of this ts state
        _sym_obs = state_obs_dd.restrict(_nxt_ts_state)

        # get the next DFA state
        _nxt_dfa = dfa_tr.restrict(curr_dfa_state & _sym_obs)
        if ADD_flag:
            _nxt_dfa = _nxt_dfa.existAbstract(reduce(lambda x, y: x & y, ts_handle.sym_add_vars_lbl))
            _nxt_dfa = _nxt_dfa.bddPattern().toADD()
        else:
            _nxt_dfa = _nxt_dfa.existAbstract(reduce(lambda x, y: x & y, ts_handle.sym_vars_lbl))
        
        # finally swap variables of TS and DFA 
        curr_ts_state = _nxt_ts_state
        curr_dfa_state = _nxt_dfa.swapVariables(dfa_curr_vars, dfa_next_vars)

        assert counter <= len(action_map), "Error while extracting a valid plan for single DFA."
        counter += 1

    return _strategy