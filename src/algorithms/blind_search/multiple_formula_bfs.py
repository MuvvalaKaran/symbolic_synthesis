from logging import warning
import re
import sys
import itertools

from typing import List, Union
from functools import reduce
import warnings

from cudd import Cudd, BDD, ADD
from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import SymbolicMultipleDFA

class MultipleFormulaBFS(BaseSymbolicSearch):
    """
    Given a Transition systenm, and n DFAs associated with different Formulas, this class computed the shorted path (in terms of # of edges taken to)
    """

    def __init__(self,
                 init_TS: BDD,
                 dfa_handle: SymbolicMultipleDFA,
                 ts_curr_vars: List[BDD],
                 ts_next_vars: List[BDD],
                 dfa_curr_vars: List[BDD],
                 dfa_next_vars: List[BDD],
                 ts_obs_vars: List[BDD],
                 ts_trans_func_list: List[BDD], 
                 ts_sym_to_curr_map: dict,
                 ts_sym_to_S2O_map: dict,
                 tr_action_idx_map: dict, 
                 state_obs_bdd: BDD,
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = init_TS
        self.dfa_handle = dfa_handle
        self.target_DFA_list = dfa_handle.sym_goal_state_list
        self.init_DFA_list = dfa_handle.sym_init_state_list
        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_transition_fun_list = ts_trans_func_list
        self.dfa_transition_fun_list = dfa_handle.dfa_bdd_tr_list
        self.ts_sym_to_curr_state_map: dict = ts_sym_to_curr_map
        self.ts_sym_to_S2obs_map: dict = ts_sym_to_S2O_map
        self.dfa_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_sym_map_curr.inv
        self.obs_bdd = state_obs_bdd
        self.tr_action_idx_map = tr_action_idx_map
        self.dfa_state_int_map: dict = dfa_handle.node_int_map_dfas
        self.dfa_predecessors_mapping: List[dict] = []
        self._create_dfa_preds()

    
    def _create_dfa_preds(self):
        """
        A function that create a mapping from a given DFA state to it predeccsors. During the Backward search,
         significant time would be spent computing the predecessors of the current DFA state.
        
        This functions loops over all the DFA state and creates their respective predecssor and store them in a dictionary.
         Since, we have multiple DFAs, the mapping corresponding to each state of the ith DFA will be stored as a dictionary
         which can be accessed at the ith index of the list.
        """
        ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)
        dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)

        for dfa_idx, dfa in enumerate(self.dfa_handle.dfa_list):
            dfa_w_no_obs = self.dfa_transition_fun_list[dfa_idx].existAbstract(ts_obs_cube)
            dfa_pred_dict = {}
            for _to_state in list(dfa._graph.nodes()):
                _sym_current_dfa = self.dfa_sym_to_curr_state_map.inv[f'{_to_state}_{dfa_idx}']
                pred_dfa = self.pre_per_action(trans_action=dfa_w_no_obs,
                                               From=_sym_current_dfa,
                                               ycube=dfa_ycube,
                                               x_list=self.dfa_x_list,
                                               y_list=self.dfa_y_list)    # this is in terms of current dfa state variables

                _sym_from_dfa_cubes = self.convert_cube_to_func(dd_func=pred_dfa, curr_state_list=self.dfa_x_list)
                for _sym_from_dfa_state in _sym_from_dfa_cubes:
                    _from_dfa_state = self.dfa_sym_to_curr_state_map[_sym_from_dfa_state]
                    _from_dfa_state = re.split('_\d', _from_dfa_state)[0]
                    assert _from_dfa_state in list(dfa._graph.nodes()), "Error while creating DFA predecessor dictionary. FIX THIS!!!"
                    if _to_state in dfa_pred_dict:
                        if not isinstance(dfa_pred_dict[_to_state], list):
                            dfa_pred_dict[_to_state] = [dfa_pred_dict[_to_state]]
                            dfa_pred_dict[_to_state].append(_from_dfa_state)
                    else:
                        dfa_pred_dict[_to_state] = _from_dfa_state
            
            self.dfa_predecessors_mapping.append(dfa_pred_dict)
    

    def compute_dfa_evolution(self, bdd_func: BDD, from_dfa_states: List[BDD], verbose: bool = False) -> dict:
        """
        A function that compute the next states on each DFA given a state or a set of states

        bdd_func: The bdd associated with the set of states of the Trasition System
        from_dfa_states: The states of DFA states where you are right now. 
        """
        _nxt_dfa_states = [self.manager.bddZero() for _ in range(len(self.dfa_transition_fun_list))]

        assert len(_nxt_dfa_states) == len(from_dfa_states), "Error computing DFA evolution. FIX THIS!!!"

        # get each individual cubes from the image of TS
        # get the observation of state(s)
        ts_cubes = self.convert_cube_to_func(dd_func=bdd_func, curr_state_list=self.ts_x_list)
        # obs_bdd = self.obs_bdd.restrict(bdd_func)
        # _states_to_be_added_to_reachlist = {key: self.manager.bddZero() for key in self.dfa_sym_to_curr_state_map.inv.keys()}
        dfa_tuple_ts_state_map = {}
        dfa_list = [0 for _ in range(len(self.dfa_transition_fun_list))]

        for ts_state in ts_cubes:
            
            for dfa_idx, dfa_tr in enumerate(self.dfa_transition_fun_list):
                # check where you evolve in each DFA
                state_obs = self.obs_bdd.restrict(ts_state)
                dfa_state = dfa_tr.restrict(state_obs & from_dfa_states[dfa_idx])
                dfa_state = dfa_state.swapVariables(self.dfa_y_list, self.dfa_x_list)
                dfa_state = re.split('_\d', self.dfa_sym_to_curr_state_map[dfa_state])[0]
                dfa_list[dfa_idx] = self.dfa_state_int_map[dfa_idx][dfa_state]
            
            dfa_tuple = tuple(dfa_list)
            if dfa_tuple in dfa_tuple_ts_state_map:
                dfa_tuple_ts_state_map[dfa_tuple] |= ts_state
            else:
                dfa_tuple_ts_state_map[dfa_tuple] = ts_state
            if verbose:
                _from_dfa_tuple = self.map_dfa_state_to_tuple(from_dfa_states)
                print(f"State {self.ts_sym_to_curr_state_map[ts_state]} caused the evolution from {_from_dfa_tuple} to {dfa_tuple}")
                
        
        return dfa_tuple_ts_state_map

    def map_dfa_state_to_tuple(self, dfa_states: List[BDD]) -> tuple:
        """
        Given a list of sybolic DFA state, create  
        """
        _to_tuple = []
        for dfa_idx, _sym_s in enumerate(dfa_states):
            _s = self.dfa_sym_to_curr_state_map[_sym_s]
            dfa_state = re.split('_\d', _s)[0]
            _to_tuple.append(self.dfa_state_int_map[dfa_idx][dfa_state])
        
        return tuple(_to_tuple)

    
    def map_dfa_tuple_to_sym_states(self, dfa_tuple: tuple) -> List[str]:
        """
        Given a tuple, this function returns a list of DFA states
        """
        _dfa_states = []
        for dfa_idx, _state_num in enumerate(dfa_tuple):
            _state_name = self.dfa_state_int_map[dfa_idx].inv[_state_num]
            _dfa_states.append(self.dfa_sym_to_curr_state_map.inv[f'{_state_name}_{dfa_idx}'])
            
            # dfa_state = re.split('_\d', _sym_s)[0]
            # return self.dfa_state_int_map[dfa_idx][dfa_state]
        return _dfa_states

    
    def add_init_state_to_reached_list(self, open_list: list, closed: dict, verbose: bool = False) -> str:
        """
        A function that checks if the initial TS state enables any transition on the DFA. If yes, then update that specific DFA state list
        Else, add the initial TS state to the initial DFA state (usually T0_Init). 
        """
        _dfa_tuple_ts_state_map = self.compute_dfa_evolution(bdd_func=self.init_TS, from_dfa_states=self.init_DFA_list, verbose=verbose)

        for _key, ts_state in _dfa_tuple_ts_state_map.items():
            open_list.append({_key: ts_state})
            closed[_key] = self.manager.bddZero()
    
    def remove_states_explored(self, layer_num: int, open_list: dict, closed: BDD):
        """
        A helper function to remove all the states that have already been explored, i.e., belong to the closed ADD function.
        """
        # if the closed set is empty, then return False. 
        for _prod_dfa in open_list[layer_num].keys():
            open_list[layer_num][_prod_dfa] = open_list[layer_num][_prod_dfa] & ~closed.get(_prod_dfa, self.manager.bddZero())
    

    def check_open_list_is_empty(self, layer_num: int, open_list: list):
        """
        A helper function to check if the reached associated with all the dfa states is empty or not
        """
        for _prod_dfa in open_list[layer_num].keys():
            if not open_list[layer_num][_prod_dfa].isZero():
                return False
        return True
    

    def check_init_in_bucket(self, bucket: dict):
        """
        A helper function to check if the reached associated with all the dfa states is empty or not
        """
        prod_init_state_tuple = self.map_dfa_state_to_tuple(self.init_DFA_list)
        # for _dfa in bucket.keys():
        if prod_init_state_tuple in bucket:
            if self.init_TS <= bucket[prod_init_state_tuple]:
                return True
        
        return False
    

    def get_states_from_dd(self, dd_func: Union[BDD, ADD], curr_state_list: list, sym_map: dict) -> None:
        """
        A function thats wraps around convert_cube_to_func() and spits out the states in the corresponding
        """
        #  extract the set of states that are being expanded during each iteration
        if isinstance(dd_func, ADD):
            tmp_dd_func: BDD = dd_func.bddPattern()
            tmp_state_list: List[BDD] = [_avar.bddPattern() for _avar in curr_state_list]
            ts_cube_string = self.convert_cube_to_func(dd_func=tmp_dd_func, curr_state_list=tmp_state_list)
        
        else:
            ts_cube_string = self.convert_cube_to_func(dd_func=dd_func, curr_state_list=curr_state_list)
        
        for _s in ts_cube_string:
            _name = sym_map.get(_s)
            assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
            print(_name)
    

    def _get_ts_states_to_dfa_evolution(self, ts_states, dfa_to_state, dfa_ycube: BDD, verify: bool = False):
        """
        A helpr function that give a set of Treansition system system, compute their respective observation.
        Then, given the next DFA state computes the set of predecessors of the DFA state along TS state, that enabled that transtion

        USe the verify flag to check correctness of the DFA support variables. It is set to False by default.

        Note: Make sure that the dfa_ to_state variables are in terms of dfa next states and
         dfa_from_states are in terms of dfa_from_state variables.
        """
        if verify:
            if isinstance(dfa_to_state, ADD):
                dfa_to_state.bddPattern().support() >= dfa_ycube.bddPattern()
            else:
                for _s in dfa_to_state:
                    # the support can contain some or all element of dfa_cubes. Thats is why >= rather than ==
                    assert _s.support() >= dfa_ycube, "Error while computing TS states for 'to-dfa-state-evolution'. \
                    Make sure your 'to-dfa-states' are in terms of next dfa variables "

        # compute all the possible Transition system states
        ts_cubes = self.convert_cube_to_func(dd_func=ts_states, curr_state_list=self.ts_x_list)
        _from_dfa_states_maps = []
        

        for _ts_state in ts_cubes:
            # get their corresponding observation
            _dfa_list = []
            _ts_state_obs = self.obs_bdd.restrict(_ts_state)
            for dfa_idx, dfa_tr in enumerate(self.dfa_transition_fun_list):
                _from_dfa_state = dfa_tr.restrict(_ts_state_obs & dfa_to_state[dfa_idx])
                # as we have sahred boolean variables among all DFA states, we could have spurious cube.
                _from_dfa_cubes = self.convert_cube_to_func(dd_func=_from_dfa_state, curr_state_list=self.dfa_x_list)

                for _sym_from_dfa_state in _from_dfa_cubes:
                    if _sym_from_dfa_state in self.dfa_sym_to_curr_state_map.keys():
                        _from_dfa_state = self.dfa_sym_to_curr_state_map[_sym_from_dfa_state]
                        _from_dfa_state = re.split('_\d', _from_dfa_state)[0]
                        try:
                            if self.dfa_state_int_map[dfa_idx][_from_dfa_state] in _dfa_list[dfa_idx]:
                                warnings.warn("Error while computing valid Pre DFA states. FIX THIS!!!!")
                            else:
                                _dfa_list[dfa_idx][self.dfa_state_int_map[dfa_idx][_from_dfa_state]] = _ts_state
                        except:
                            # _dfa_list.append([self.dfa_state_int_map[dfa_idx][_from_dfa_state]])
                            _dfa_list.append({self.dfa_state_int_map[dfa_idx][_from_dfa_state] : _ts_state})

            _from_dfa_states_maps.append(_dfa_list)
        
        _dfa_ind_keys = [list(_d.keys()) for _d in _from_dfa_states_maps]
        # now that we have stored all possible pre dfa states for each dfa, we will construct all possible combinations
        _dfa_tuples = list(itertools.product(*_dfa_ind_keys))
        _prod_dfa_state_ts_map = {}
        for _idx, _t in enumerate(_dfa_tuples):
            if _t in _prod_dfa_state_ts_map:
                _prod_dfa_state_ts_map[_t] |= _from_dfa_states_maps[_idx][_t]
            else:
                _prod_dfa_state_ts_map[_t] = _from_dfa_states_maps[_idx][_t]
                    
        return _prod_dfa_state_ts_map
    

    def updated_closed_list(self, closed: dict, bucket: BDD) -> dict:
        """
        Update the closed set by iterating over the entire bucket (all DFA states) reached list,
         and adding them to the expanded set (closed set)
        """
        for _prod_dfa_state, _ts_states in bucket.items():
            # if a prod_dfa closed set already exists then take disjunction, else intialize one
            if _prod_dfa_state in closed:
                closed[_prod_dfa_state] |= _ts_states
            else:
                closed[_prod_dfa_state] = _ts_states
        
        return closed
    

    def symbolic_bfs_nLTL(self, verbose: bool = False):
        """
        Implement a general BFS planning algorithm over multiple formulas in a symbolic fashion.

        """
        # while open list is a sequence of bucket closed is a just a set of explored states and hence is not numbered
        open_list = []
        closed = {}

        ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        g_layer = 0

        # add the init state to its respective DFA state. Note, we start could start in some other state 
        self.add_init_state_to_reached_list(open_list, closed, verbose=verbose)

        prod_accp_state_tuple = self.map_dfa_state_to_tuple(self.target_DFA_list)

        while not prod_accp_state_tuple in open_list[g_layer]:
            # remove all states that have been explored
            self.remove_states_explored(layer_num=g_layer, open_list=open_list, closed=closed)

            # If unexpanded states exist ... 
            if not self.check_open_list_is_empty(layer_num=g_layer, open_list=open_list):
                # Add states to be expanded next to already expanded states
                closed = self.updated_closed_list(closed, open_list[g_layer])
                
                # compute the image of the TS states 
                for prod_dfa_tuple, ts_states in open_list[g_layer].items():
                    ts_image_bdd = self.manager.bddZero()
                    for tr_action in self.ts_transition_fun_list:
                        image_c = self.image_per_action(trans_action=tr_action,
                                                        From=ts_states,
                                                        xcube=ts_xcube,
                                                        x_list=self.ts_x_list,
                                                        y_list=self.ts_y_list)
                        ts_image_bdd |= image_c
                    
                    # check where you evolve on the DFA
                    _dfa_state_tuple = self.map_dfa_tuple_to_sym_states(prod_dfa_tuple)
                    _dfa_tuple_nxt_ts_map = self.compute_dfa_evolution(bdd_func=ts_image_bdd, from_dfa_states=_dfa_state_tuple, verbose=verbose)

                    for _key, _nxt_ts_state in _dfa_tuple_nxt_ts_map.items():
                        try:
                            # if the bucket exisits thrn check if this prod dfa has been explored before or not
                            if _key in open_list[g_layer + 1]: 
                                open_list[g_layer + 1][_key] |= _nxt_ts_state
                            else:
                                open_list[g_layer + 1][_key] = _nxt_ts_state
                        except:
                            # initialize the bucket
                            open_list.append({_key: _nxt_ts_state})
            else:
                print("No plan exists! Terminating algorithm.")
                sys.exit(-1)
            
            g_layer += 1
        
        print("Found a plan, Now retireving it!")

        return self.retrieve_symbolic_bfs_nLTL_plan(freach_list=open_list,
                                                    max_layer=g_layer,
                                                    verbose=verbose)
    

    def retrieve_symbolic_bfs_nLTL_plan(self, freach_list, max_layer: int, verbose: bool = False):
        """
        Retrieve the plan from symbolic BFS algorithm for multiple DFAs. The list is sequence of composed states of TS and DFAs.
        """
        ts_ycube = reduce(lambda a, b: a & b, self.ts_y_list)
        dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)
        prod_accp_state_tuple = self.map_dfa_state_to_tuple(self.target_DFA_list)
        current_ts_dict = {prod_accp_state_tuple: freach_list[max_layer][prod_accp_state_tuple]}
        g_layer = max_layer

        parent_plan = {}

        while not self.check_init_in_bucket(current_ts_dict):
            new_current_ts: dict = {}
            for _to_dfa_states, current_ts in current_ts_dict.items():
                if current_ts.isZero():
                        continue
                preds_ts_list = []
                sym_current_dfa_states = self.map_dfa_tuple_to_sym_states(_to_dfa_states)
                for tr_num, tran_func_action in enumerate(self.ts_transition_fun_list):
                    preds_ts = self.pre_per_action(trans_action=tran_func_action,
                                                From=current_ts,
                                                ycube=ts_ycube,
                                                x_list=self.ts_x_list,
                                                y_list=self.ts_y_list)
                    preds_ts_list.append(preds_ts)
                if verbose:
                    self.get_states_from_dd(dd_func=reduce(lambda a, b: a | b, preds_ts_list), curr_state_list=self.ts_x_list, sym_map=self.ts_sym_to_curr_state_map)

                # compute the valid state predecessors on the DFAs from which we could have transited to the current DFA states given by _to_dfa_states
                valid_pre_dfa_state = self._get_ts_states_to_dfa_evolution(ts_states=current_ts,
                                                                           dfa_to_state=[_to_dfa_state.swapVariables(self.dfa_x_list, self.dfa_y_list) for _to_dfa_state in sym_current_dfa_states] ,
                                                                           dfa_ycube=dfa_ycube,
                                                                           verify=False)
                
                # compute the intersetion of
                for _from_prod_dfa_state, _v in valid_pre_dfa_state.items():
                    if not _v.isZero():
                        # check the intersection of the corresponding layer's corresponding DFA state with preds_ts_list
                        
                        for tr_num, pred_ts_bdd in enumerate(preds_ts_list):
                            try:
                                intersection: ADD = freach_list[g_layer - 1][_from_prod_dfa_state] & pred_ts_bdd
                            except:
                                if verbose:
                                    print(f"The Forward reach list dose not contain {_from_prod_dfa_state} in bucket {g_layer - 1}.\
                                        For now skipping computing this intersection. This may cause issues in the future.")
                                continue

                            if not intersection.isZero():
                                for _ts_state in self.convert_add_cube_to_func(dd_func=intersection, curr_state_list=self.ts_x_list):
                                    self._append_dict_value(dict_obj=parent_plan,
                                                            key_dfa=_from_prod_dfa_state,
                                                            key_ts=self.ts_add_sym_to_curr_state_map[_ts_state],
                                                            value=self.tr_action_idx_map.inv[tr_num])
                                
                                new_current_ts[_from_prod_dfa_state] |= intersection
            
            current_ts_dict = new_current_ts
            # sym_current_dfa = self.dfa_add_sym_to_curr_state_map.inv[_dfa_state]
            g_layer -= 1

        return parent_plan




