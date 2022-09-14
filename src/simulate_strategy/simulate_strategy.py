import re
import itertools
import warnings

import  src.gridworld_visualizer.gridworld_vis.gridworld as gridworld_handle

from typing import List
from functools import reduce
from config import *
from cudd import Cudd, BDD, ADD

from src.algorithms.base.base_symbolic_search import BaseSymbolicSearch



PDDL_TO_GRIDWORLD_MAP = {'moveright': gridworld_handle.E,
                         'moveleft': gridworld_handle.W,
                         'moveup': gridworld_handle.N,
                         'movedown': gridworld_handle.S}



def convert_action_dict_to_gridworld_strategy_nLTL(action_map: dict,
                                                transition_sys_tr,
                                                tr_action_idx_map,
                                                dfa_handle,
                                                init_state_ts,
                                                init_state_dfa_list,
                                                target_DFA_list,
                                                state_obs_dd,
                                                ts_curr_vars: list,
                                                ts_next_vars: list,
                                                dfa_curr_vars: list,
                                                dfa_next_vars: list,
                                                ts_sym_to_curr_map,
                                                dfa_sym_to_curr_map):
    """
    A DFA search over our dictionary of actions.
    """
    curr_ts_state = init_state_ts
    curr_dfa_state_tuple = map_dfa_state_to_tuple(init_state_dfa_list,
                                                    dfa_sym_to_curr_state_map=dfa_handle.dfa_predicate_sym_map_curr.inv,
                                                    dfa_state_int_map=dfa_handle.node_int_map_dfas)
    target_dfa_state_tuple = map_dfa_state_to_tuple(target_DFA_list,
                                                    dfa_sym_to_curr_state_map=dfa_handle.dfa_predicate_sym_map_curr.inv,
                                                    dfa_state_int_map=dfa_handle.node_int_map_dfas)
    



def get_dfa_evolution(dfa_handle, _nxt_ts_state, state_obs_dd, dfa_curr_vars, dfa_next_vars, curr_dfa_state_tuple):
    """
    A function to compute the next state of the DFA when evolving over the product of multiple tasks
    """
    
    # get the observation of this ts state
    # _sym_obs = state_obs_dd.restrict(_nxt_ts_state)
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


def convert_action_dict_to_gridworld_strategy_nLTL(action_map: dict,
                                                transition_sys_tr,
                                                tr_action_idx_map,
                                                dfa_handle,
                                                init_state_ts_sym,
                                                init_state_dfa_list,
                                                target_DFA_list,
                                                state_obs_dd,
                                                ts_curr_vars: list,
                                                ts_next_vars: list,
                                                dfa_curr_vars: list,
                                                dfa_next_vars: list,
                                                ts_sym_to_curr_map) -> List:
    """
    A method that extract a the shortest path from the BFS dictionary computed by the nLTL BFS algorithm and then translated it 
    to Gridworld action policy and simulated.
    """    
    curr_ts_state_sym = init_state_ts_sym
    curr_ts_state = ts_sym_to_curr_map[curr_ts_state_sym]

    curr_dfa_state_tuple = map_dfa_state_to_tuple(init_state_dfa_list,
                                                    dfa_sym_to_curr_state_map=dfa_handle.dfa_predicate_sym_map_curr.inv,
                                                    dfa_state_int_map=dfa_handle.node_int_map_dfas)
    target_dfa_state_tuple = map_dfa_state_to_tuple(target_DFA_list,
                                                    dfa_sym_to_curr_state_map=dfa_handle.dfa_predicate_sym_map_curr.inv,
                                                    dfa_state_int_map=dfa_handle.node_int_map_dfas)
    
    open_list = []
    plan_list = []
    closed_list = []

    counter = 1
    while not target_dfa_state_tuple == curr_dfa_state_tuple:
        assert 0 < counter <= len(action_map.keys()), "Error while simulating the strategy. FIX THIS!!!"
        if curr_dfa_state_tuple in action_map[counter] and curr_ts_state in action_map[counter][curr_dfa_state_tuple]:
            # if ts_sym_to_curr_map[curr_ts_state] in action_map[counter][curr_dfa_state_tuple]:
            _actions = action_map[counter][curr_dfa_state_tuple][curr_ts_state]

                # _actions.extend([ts_sym_to_curr_map[curr_ts_state]])
            if isinstance(_actions, list):
                [open_list.append((_state_a)) for _state_a in  list(itertools.product([curr_ts_state], _actions, [curr_dfa_state_tuple]))]
            else:
                [open_list.append((_state_a)) for _state_a in  list(itertools.product([curr_ts_state], [_actions], [curr_dfa_state_tuple]))]

            # # remove elements in open list that were already explored:
            # open_list = [x for x in open_list if x not in closed_list]

        else:
            # pop element from the closed list
            curr_dfa_state_tuple, curr_ts_state, _a = plan_list.pop()
            # closed_list.append((curr_ts_state, _, curr_dfa_state_tuple))
            counter -= 1
            if len(open_list) > 0:
                while not (curr_dfa_state_tuple in open_list[-1] and curr_ts_state in open_list[-1]):
                    curr_dfa_state_tuple, curr_ts_state, _a = plan_list.pop()
                    # closed_list.append((curr_ts_state, _, curr_dfa_state_tuple))
                    counter -= 1
            else:
                # since we do have the open list to check from, we check if the current DFA state exists in the
                if not (curr_dfa_state_tuple in action_map[counter] and curr_ts_state in action_map[counter][curr_dfa_state_tuple]):
                     # pop element from the closed list
                    curr_dfa_state_tuple, curr_ts_state, _a = plan_list.pop()
                    # closed_list.append((curr_ts_state, _, curr_dfa_state_tuple))
                    counter -= 1


        # pop the top mode node, expand it and repeat
        if len(open_list) > 0:
            curr_ts_state , _a, curr_dfa_state_tuple = open_list.pop()
            while (curr_ts_state , _a, curr_dfa_state_tuple) in closed_list:
                curr_ts_state , _a, curr_dfa_state_tuple = open_list.pop()
                counter -=1

        # add it to the closed list
        closed_list.append((curr_ts_state, _a, curr_dfa_state_tuple))

        # if len(open_list) == 0:
        #     if (curr_ts_state , _a, curr_dfa_state_tuple) not in closed_list:
        #         plan_list.append((curr_dfa_state_tuple, curr_ts_state , _a))
        # else:
        plan_list.append((curr_dfa_state_tuple, curr_ts_state , _a))

        _nxt_ts_state_sym = transition_sys_tr[tr_action_idx_map[_a]].restrict(ts_sym_to_curr_map.inv[curr_ts_state])
        _nxt_ts_state_sym = _nxt_ts_state_sym.swapVariables(ts_curr_vars, ts_next_vars)

        curr_dfa_state_tuple = get_dfa_evolution(dfa_handle=dfa_handle,
                                                 _nxt_ts_state=_nxt_ts_state_sym,
                                                 curr_dfa_state_tuple=curr_dfa_state_tuple,
                                                 state_obs_dd=state_obs_dd,
                                                 dfa_curr_vars=dfa_curr_vars,
                                                 dfa_next_vars=dfa_next_vars)
        
        
        curr_ts_state_sym = _nxt_ts_state_sym
        curr_ts_state = ts_sym_to_curr_map[curr_ts_state_sym]

        counter += 1
    
    return retrieve_plan(plan_list)

def retrieve_plan(plan_list: List[tuple]) -> List:
    """
    A function that compute a sequence of gridworld action, E, W, N, S
    """
    _strategy = []
    for _plan in plan_list:
        _strategy.append(PDDL_TO_GRIDWORLD_MAP[_plan[-1]])
    
    return _strategy


def create_gridworld(size: int, strategy: list, init_pos: tuple = (1, 1)):
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

    file_name = PROJECT_ROOT + f'/plots/simulated_strategy.svg'

    if strategy is None:
        warnings.warn("Strategy is None. Unrolling Default strategy")
        strategy = [gridworld_handle.E, gridworld_handle.N, gridworld_handle.N,
                    gridworld_handle.N, gridworld_handle.N, gridworld_handle.W, 
                    gridworld_handle.W, gridworld_handle.W]
    if OBSTACLE:
        svg = gridworld_handle.gridworld(n=size, tile2classes=tile2classes_obstacle, actions=strategy, init_pos=init_pos)
    else:
        svg = gridworld_handle.gridworld(n=size, tile2classes=tile2classes, actions=strategy, init_pos=init_pos)
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


def map_dfa_state_to_tuple_no_list(dfa_states: List[BDD], dfa_sym_to_curr_state_map: dict, dfa_state_int_map: dict) -> tuple:
    """
    Given a list of sybolic DFA state, create  
    """
    _to_tuple = [None for _ in dfa_state_int_map.keys()]
    for _sym_s in dfa_states:
        if _sym_s in dfa_sym_to_curr_state_map.keys():
            _s = dfa_sym_to_curr_state_map[_sym_s]
            dfa_state = re.split('_\d', _s)[0]
            dfa_idx = int(re.split('_', _s)[-1])
            _to_tuple[dfa_idx] = dfa_state_int_map[dfa_idx][dfa_state]
    
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


def convert_action_dict_to_gridworld_strategy(action_map: dict,
                                              transition_sys_tr,
                                              tr_action_idx_map,
                                              dfa_tr,
                                              init_state_ts,
                                              init_state_dfa,
                                              target_DFA,
                                              state_obs_dd,
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
    
    _strategy = []

    curr_ts_state = init_state_ts
    # get the observation of the initial state
    # obs_dd = state_obs_dd.restrict(init_state_ts)

    # check if any of the DFA edges are satisfied
    # image_DFA = dfa_tr.restrict(init_state_dfa & obs_dd)
    # image_DFA = image_DFA.swapVariables(dfa_next_vars, dfa_curr_vars)
    # _explicit_dfa_state: str = dfa_sym_to_curr_map[image_DFA] 


    curr_dfa_state = init_state_dfa

    while not target_DFA == curr_dfa_state:
        # get the strategy
        try:
            _a = action_map[dfa_sym_to_curr_map[curr_dfa_state]][ts_sym_to_curr_map[curr_ts_state]]
        except:
            return _strategy

        try:
            _strategy.append(PDDL_TO_GRIDWORLD_MAP.get(_a))
        except:
            _strategy.append(PDDL_TO_GRIDWORLD_MAP.get(_a[0]))

        # get the next TS
        try:
            _nxt_ts_state = transition_sys_tr[tr_action_idx_map[_a]].restrict(curr_ts_state)
        except:
            _nxt_ts_state = transition_sys_tr[tr_action_idx_map[_a[0]]].restrict(curr_ts_state)
        
        _nxt_ts_state = _nxt_ts_state.swapVariables(ts_curr_vars, ts_next_vars)
        
        # get the observation of this ts state
        _sym_obs = state_obs_dd.restrict(_nxt_ts_state)

        # get the next DFA state
        _nxt_dfa = dfa_tr.restrict(curr_dfa_state & _sym_obs)
        
        # finally swap variables of TS and DFA 
        curr_ts_state = _nxt_ts_state
        curr_dfa_state = _nxt_dfa.swapVariables(dfa_curr_vars, dfa_next_vars)

    return _strategy
