import re
import sys
import warnings
import itertools

from functools import reduce
from typing import List, Union


from cudd import Cudd, BDD, ADD
from src.symbolic_graphs import SymbolicMultipleDFA, SymbolicWeightedTransitionSystem
from src.algorithms.base import BaseSymbolicSearch


class MultipleFormulaDijkstra(BaseSymbolicSearch):
    """
    Given a Transition systenm, and n DFAs associated with different Formulas, this class computes the minimum cost path
    """
    def __init__(self,
                 ts_handle: SymbolicWeightedTransitionSystem,
                 dfa_handle: SymbolicMultipleDFA,
                 ts_curr_vars: List[ADD],
                 ts_next_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 dfa_next_vars: List[ADD],
                 ts_obs_vars: list,
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle
        self.init_TS = ts_handle.sym_add_init_states
        self.target_DFA_list = dfa_handle.sym_goal_state_list
        self.init_DFA_list = dfa_handle.sym_init_state_list
        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars

        self.ts_transition_fun_list = ts_handle.sym_tr_actions
        self.dfa_monolithic_tr_func = reduce(lambda a, b: a | b,  dfa_handle.dfa_add_tr_list)
        self.dfa_transition_fun_list = dfa_handle.dfa_add_tr_list

        self.ts_add_sym_to_curr_state_map: dict = ts_handle.predicate_add_sym_map_curr.inv
        self.ts_bdd_sym_to_curr_state_map: dict = ts_handle.predicate_sym_map_curr.inv
        self.ts_bdd_sym_to_S2obs_map: dict =  ts_handle.predicate_sym_map_lbl.inv
        self.ts_add_sym_to_S2obs_map: dict =  ts_handle.predicate_add_sym_map_lbl.inv
        self.dfa_bdd_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_sym_map_curr.inv
        self.dfa_add_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_add_sym_map_curr.inv
        self.dfa_state_int_map: dict = dfa_handle.node_int_map_dfas

        self.obs_add: ADD = ts_handle.sym_add_state_labels
        self.tr_action_idx_map: dict = ts_handle.tr_action_idx_map
    

    def add_init_ADD_state_to_reached_list(self, open_list: dict, closed: dict, verbose: bool = False) -> str:
        """
        A function that checks if the initial TS state enables any transition on the DFA. If yes, then update that specific DFA state list
        Else, add the initial TS state to the initial DFA state (usually T0_Init). 
        """
        _dfa_tuple_ts_state_map = self.compute_ADD_dfa_evolution(add_func=self.init_TS, from_dfa_states=self.init_DFA_list, verbose=verbose)

        _prod_init_state_tuple = self.map_dfa_state_to_tuple(self.init_DFA_list)

        assert len(_dfa_tuple_ts_state_map.keys()) == 1, "The initial state triggered multiple states in one of the DFA. FIX THIS!!!!"
        if _prod_init_state_tuple != list(_dfa_tuple_ts_state_map.keys())[0]:
            print("*******************************************************************")
            warnings.warn("Looks like the initial state the robot is starting in, triggered at least one of DFA's evolution.")
            print("*******************************************************************")

        for _key, ts_state in _dfa_tuple_ts_state_map.items():
            _key: tuple = _prod_init_state_tuple
            # append the init state to the 0th layer
            open_list[0] = {_key: ts_state}
            closed[_key] = self.manager.addZero()
    

    def compute_ADD_dfa_evolution(self, add_func: BDD, from_dfa_states: List[ADD], verbose: bool = False) -> dict:
        """
        A function that compute the next states on each DFA given a state or a set of states

        add_func: The add associated with the set of states of the Trasition System
        from_dfa_states: The states of DFA states where you are right now. 
        """
        _nxt_dfa_states = [self.manager.addZero() for _ in range(len(self.dfa_transition_fun_list))]

        assert len(_nxt_dfa_states) == len(from_dfa_states), "Error computing DFA evolution. FIX THIS!!!"

        # get each individual cubes from the image of TS
        # get the observation of state(s)
        ts_cubes: List[ADD] = self.convert_add_cube_to_func(dd_func=add_func, curr_state_list=self.ts_x_list)
        dfa_tuple_ts_state_map = {}
        dfa_list = [0 for _ in range(len(self.dfa_transition_fun_list))]

        for ts_state in ts_cubes:
            
            for dfa_idx, dfa_tr in enumerate(self.dfa_transition_fun_list):
                # check where you evolve in each DFA
                state_obs = self.obs_add.restrict(ts_state)
                dfa_state: ADD = dfa_tr.restrict(state_obs & from_dfa_states[dfa_idx])
                dfa_state: ADD = dfa_state.swapVariables(self.dfa_y_list, self.dfa_x_list)
                dfa_state: str = re.split('_\d', self.dfa_add_sym_to_curr_state_map[dfa_state])[0]
                dfa_list[dfa_idx] = self.dfa_state_int_map[dfa_idx][dfa_state]
            
            dfa_tuple = tuple(dfa_list)
            if dfa_tuple in dfa_tuple_ts_state_map:
                dfa_tuple_ts_state_map[dfa_tuple] |= ts_state
            else:
                dfa_tuple_ts_state_map[dfa_tuple] = ts_state
            if verbose:
                _from_dfa_tuple = self.map_dfa_state_to_tuple(from_dfa_states)
                print(f"State {self.ts_add_sym_to_curr_state_map[ts_state]} caused the evolution from {_from_dfa_tuple} to {dfa_tuple}")
                
        
        return dfa_tuple_ts_state_map
    
    # NOTE: This function can be added to the base class
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
        
    
    def _get_max_tr_action_cost(self) -> int:
        """
        A helper function that retireves the highest cost amongst all the transiton function costs
        """
        _max = 0
        for tr_action in self.ts_transition_fun_list:
                action_cost = tr_action.findMax()
                action_cost_int = int(re.findall(r'\d+', action_cost.__repr__())[0])
                if action_cost_int > _max:
                    _max = action_cost_int
        
        return _max

    
    def _print_freach_list(self, open_list: dict):
        """
        A helper method to print the Forward reach set
        """
        for layer, bucket in open_list.items():
            print(f"******************Currently in Bucket {layer}******************")
            for _dfa_state, _ts_state in bucket.items(): 
                _ts_cubes = self.convert_add_cube_to_func(_ts_state, self.ts_x_list)
                print(f"{_dfa_state}: Reached State {[self.ts_add_sym_to_curr_state_map[_ts_state ] for _ts_state in _ts_cubes]}")

    
    def map_dfa_state_to_tuple(self, dfa_states: List[ADD]) -> tuple:
        """
        Given a list of sybolic DFA state, create  
        """
        _to_tuple = []
        for dfa_idx, _sym_s in enumerate(dfa_states):
            _s = self.dfa_add_sym_to_curr_state_map[_sym_s]
            dfa_state = re.split('_\d', _s)[0]
            _to_tuple.append(self.dfa_state_int_map[dfa_idx][dfa_state])
        
        return tuple(_to_tuple)
    

    def map_dfa_tuple_to_sym_states(self, dfa_tuple: tuple) -> List[ADD]:
        """
        Given a tuple, this function returns a list of DFA states
        """
        _dfa_states = []
        for dfa_idx, _state_num in enumerate(dfa_tuple):
            _state_name = self.dfa_state_int_map[dfa_idx].inv[_state_num]
            _dfa_states.append(self.dfa_add_sym_to_curr_state_map.inv[f'{_state_name}_{dfa_idx}'])
        return _dfa_states


    # NOTE: This function can be added to the base class
    def remove_states_explored(self, layer_num: int, open_list: dict, closed: ADD):
        """
        A helper function to remove all the states that have already been explored, i.e., belong to the closed ADD function.
        """
        # if the closed set is empty, then return False. 
        for _prod_dfa in open_list[layer_num].keys():
            open_list[layer_num][_prod_dfa] = open_list[layer_num][_prod_dfa] & ~closed.get(_prod_dfa, self.manager.addZero())

    # NOTE: This function can be added to the base class
    def updated_closed_list(self, closed: dict, bucket: dict, verbose: bool = False) -> dict:
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
            if verbose:
                    print(f"Updated closed list of {_prod_dfa_state} with following states:")
                    self.get_states_from_dd(dd_func=_ts_states,
                                            curr_state_list=self.ts_x_list,
                                            sym_map=self.ts_bdd_sym_to_curr_state_map)
        return closed

    # NOTE: This function can be added to the base class
    def check_open_list_is_empty(self, layer_num: int, open_list: dict):
        """
        A helper function to check if the reached associated with all the dfa states is empty or not
        """
        for _dfa in open_list[layer_num].keys():
            if not open_list[layer_num][_dfa].isZero():
                return False
        
        return True
    
    # NOTE: This function can be added to the base class
    def check_init_in_bucket(self, bucket: dict):
        """
        A helper function to check if the reached associated with all the dfa states is empty or not
        """
        prod_init_state_tuple = self.map_dfa_state_to_tuple(self.init_DFA_list)
        if prod_init_state_tuple in bucket:
            if self.init_TS <= bucket[prod_init_state_tuple]:
                return True
        
        return False
    

    def check_from_to_dfa_evolution_validity(self,
                                             from_dfa_state_tuple,
                                             to_dfa_state_tuple,
                                             ts_states, ts_xcube,
                                             ts_ycube, ts_obs_cube,
                                             ts_transition_fun) -> ADD:
        """
        A function to check if given a set of TS states, to and from DFA state, if any or all of the state enable that transition.
        
        Return the states that do enable the transition.   
        """
        # check if this intersection actually enables the transition _from_prod_dfa_state to _to_dfa_States
        image_state_act = self.image_per_action(trans_action=ts_transition_fun,
                                                From=ts_states,
                                                xcube=ts_xcube,
                                                x_list=self.ts_x_list,
                                                y_list=self.ts_y_list)

        _sym_from_dfa_state: List[ADD] = self.map_dfa_tuple_to_sym_states(from_dfa_state_tuple)
        _sym_to_dfa_state: List[ADD] = self.map_dfa_tuple_to_sym_states(to_dfa_state_tuple)

        # get the corresponding state observation
        _ts_obs = self.obs_add.restrict(image_state_act)

        _actual_obs = self.manager.addOne()  
        for _dfa_idx, (_sym_from_dfa, _sym_to_dfa) in enumerate(zip(_sym_from_dfa_state, _sym_to_dfa_state)):
            _possible_valid_obs_bdd = self.dfa_transition_fun_list[_dfa_idx].restrict(_sym_from_dfa & _sym_to_dfa.swapVariables(self.dfa_x_list, self.dfa_y_list) & _ts_obs)
            _actual_obs = _actual_obs & _possible_valid_obs_bdd
        
        # if all the states enable the transition from (x, y, z) ---> (x', y', z') then return the whole set
        if _actual_obs.isOne():
            return ts_states
        # if even one the transitions in the prod DFA state fails the transition from (x, y, z) ---> (x', y', z') then return the empty set
        elif _actual_obs.isZero():
            return self.manager.addZero() 
        # its neither True nor False, meaning a subset of state enable the transtion 
        else:
            _actual_ts = self.obs_add.restrict(_actual_obs).existAbstract(ts_obs_cube)
            # compute the pre and take the intersection with the current ts_states
            pred_ts = self.pre_per_action(trans_action=ts_transition_fun,
                                          From=_actual_ts,
                                          ycube=ts_ycube,
                                          x_list=self.ts_x_list,
                                          y_list=self.ts_y_list)
            _valid_intersection = pred_ts & ts_states
            assert not _valid_intersection.isZero() , f"Error while computing valid pre TS states transtioning from {from_dfa_state_tuple} to {to_dfa_state_tuple}. FIX THIS!!!!!"
            return _valid_intersection
    

    def compute_image_of_composed_graph(self,
                                        open_list,
                                        g_layer: int,
                                        g_val: ADD,
                                        ts_xcube: ADD,
                                        verbose: bool = False):
        # compute the image of the TS states 
        for from_dfa_tuple, ts_states in open_list[g_layer].items():
            for tr_action in self.ts_transition_fun_list:
                # ts_image_bdd = self.manager.bddZero()
                # first get the corresponding transition action cost (constant at the terminal node)
                action_cost = tr_action.findMax()
                step = g_val + action_cost
                step_val = int(re.findall(r'\d+', step.__repr__())[0])

                ts_image_add = self.image_per_action(trans_action=tr_action,
                                                     From=ts_states,
                                                     xcube=ts_xcube,
                                                     x_list=self.ts_x_list,
                                                     y_list=self.ts_y_list)

                if ts_image_add.isZero():
                    continue
            
                if verbose:
                    self.get_states_from_dd(dd_func=ts_image_add,
                                            curr_state_list=self.ts_x_list,
                                            sym_map=self.ts_bdd_sym_to_curr_state_map)
                
                # check where you evolve on the DFA
                _dfa_state_tuple = self.map_dfa_tuple_to_sym_states(from_dfa_tuple)
                _dfa_tuple_nxt_ts_map = self.compute_ADD_dfa_evolution(add_func=ts_image_add,
                                                                       from_dfa_states=_dfa_state_tuple,
                                                                       verbose=verbose)
                
                for _key, _nxt_ts_state in _dfa_tuple_nxt_ts_map.items():
                    try:
                        # if the bucket exists then check if this prod dfa has been explored before or not
                        if _key in open_list[step_val]: 
                            open_list[step_val][_key] |= _nxt_ts_state
                        else:
                            open_list[step_val][_key] = _nxt_ts_state
                    except:
                        # initialize the bucket
                        open_list[step_val] = {_key: _nxt_ts_state}


    def _get_pred_prod_dfa_ts_map(self,
                                  ts_obs_cube: BDD,
                                  sym_dfa_to_states: List[ADD],
                                  dfa_ycube: ADD,
                                  verify: bool = False):
        """
        A function to compute the valid pre Prod DFA states uses in the backward search algorithm.

        ts_state: Current States we are at in the Transition System
        dfa_to_states: The current dfa states from which we want to compute the valid pre dfa states.

        Idea: we want to check under which DFA states, were we able to transition the current set of DFA states for a given set of TS states.
        """
        if verify:
            if isinstance(sym_dfa_to_states, ADD):
                sym_dfa_to_states.bddPattern().support() >= dfa_ycube.bddPattern()
            else:
                for _s in sym_dfa_to_states:
                    # the support can contain some or all element of dfa_cubes. Thats is why >= rather than ==
                    assert _s.support() >= dfa_ycube, "Error while computing TS states for 'to-dfa-state-evolution'. \
                    Make sure your 'to-dfa-states' are in terms of next dfa variables "

        dfa_w_no_obs = self.dfa_monolithic_tr_func.existAbstract(ts_obs_cube)
        _sym_dfa_to_states_bdd = reduce(lambda a, b: a | b, sym_dfa_to_states)

        _pre_dfa = self.pre_per_action(trans_action=dfa_w_no_obs,
                                       From=_sym_dfa_to_states_bdd,
                                       ycube=dfa_ycube,
                                       x_list=self.dfa_x_list,
                                       y_list=self.dfa_y_list)

        # enumerate the set of possible DFAs into their corresponding DFA idx 
        _pre_dfa_cubes = self.convert_add_cube_to_func(dd_func=_pre_dfa, curr_state_list=self.dfa_x_list)
        _dfa_list = [[] for _ in range(len(self.dfa_transition_fun_list))]
        for _sym_from_dfa_state in _pre_dfa_cubes:
            if _sym_from_dfa_state in self.dfa_add_sym_to_curr_state_map.keys():
                _full_from_dfa_state = self.dfa_add_sym_to_curr_state_map[_sym_from_dfa_state]
                _from_dfa_state = re.split('_\d', _full_from_dfa_state)[0]
                _dfa_idx = int(re.split('_', _full_from_dfa_state)[-1])
                _dfa_list[_dfa_idx].append(_from_dfa_state)
        
        # compute all possible combinations of valid pre dfa states
        _valid_pre_dfa_state_combos = list(itertools.product(*_dfa_list))

        # not all the prod states are physically possible on the actually giant product automaton. We need to remove such edges. 

        _pre_prod_dfa_ts_map = {}        

        for _valid_pre in _valid_pre_dfa_state_combos:
            _sym_pre_dfa_states = [self.dfa_add_sym_to_curr_state_map.inv[f'{_s}_{_idx}'] for _idx, _s in enumerate(_valid_pre)]
            _actual_obs = self.manager.addOne()            
            # check if there exists a transition individually and check if there is a valid intersection of all.
            for _dfa_idx, _sym_pre_dfa in enumerate(_sym_pre_dfa_states):
                _possible_valid_obs_dd = self.dfa_transition_fun_list[_dfa_idx].restrict(_sym_pre_dfa & sym_dfa_to_states[_dfa_idx].swapVariables(self.dfa_x_list, self.dfa_y_list))
                _actual_obs = _actual_obs & _possible_valid_obs_dd
            
            _pre_prod_tuple = self.map_dfa_state_to_tuple(_sym_pre_dfa_states)
            _pre_prod_dfa_ts_map[_pre_prod_tuple] = _actual_obs
        
        return _pre_prod_dfa_ts_map



    def symbolic_dijkstra_nLTL(self, verbose: bool = False) -> dict:
        """
        This function implements a Dijkstra's algorithm as outline in Peter Kissmann Ph.D. Thesis. 

        We implement bucket based approach to expand and store the search frontier.

        Todo: Write code to check if all the edge weights are or not:
        if yes, check if they are all equal to one: then just multiple_formula_bfs() as that code uses BDD
         which are inherently faster
            else: TODO: We need to write code in which we do not loop over each TR individually, but as a whole. 
        else:
            we run the current implementation we have. 
        """

        # while open list is a sequence of bucket closed is a just a set of explored states and hence is not numbered
        # unlike BFS algorithm, open list here is a dictionary as we can jump artirarily from one bucket to another based on weights
        # The layer number is thus stored as key and BDDs as the value associated with it.
        open_list = {}
        closed = {}
        c_max = self._get_max_tr_action_cost()
        empty_bucket_counter: int = 0

        ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        g_val = self.manager.addZero()
        if g_val.isZero():
            g_layer = 0

        # add the init state to its respective DFA state. Note, we start could start in some other state 
        self.add_init_ADD_state_to_reached_list(open_list, closed, verbose=verbose)

        prod_accp_state_tuple = self.map_dfa_state_to_tuple(self.target_DFA_list)

        while not prod_accp_state_tuple in open_list[g_layer]:
            # remove all states that have been explored
            self.remove_states_explored(layer_num=g_layer, open_list=open_list, closed=closed)

            # If unexpanded states exist ... 
            if not self.check_open_list_is_empty(layer_num=g_layer, open_list=open_list):
                # reset the empty bucket counter 
                empty_bucket_counter = 0
                # Add states to be expanded next to already expanded states
                closed = self.updated_closed_list(closed, open_list[g_layer], verbose=False)

                # Find successors and add them to their respectice buckets
                self.compute_image_of_composed_graph(open_list=open_list,
                                                     g_layer=g_layer,
                                                     g_val=g_val,
                                                     ts_xcube=ts_xcube,
                                                     verbose=verbose)
            
            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == c_max:
                    print("No plan exists! Terminating algorithm.")
                    sys.exit(-1)
            
            g_val = g_val + self.manager.addOne()
            g_layer += 1
            
            # keep updating g_layer up until the most recent bucket
            while g_layer not in open_list:
                g_val = g_val + self.manager.addOne()
                g_layer += 1

        print(f"Found a plan with least cost lenght {g_layer}, Now retireving it!")

        if verbose:
            self._print_freach_list(open_list=open_list)
        
        return self.retrieve_symbolic_dijkstra_nLTL_plan(freach_list=open_list,
                                                         max_layer=g_layer,
                                                         verbose=verbose)
        
    
    def retrieve_symbolic_dijkstra_nLTL_plan(self, freach_list: dict, max_layer: int, verbose: bool = False) -> dict:
        """
        A function to retrieve the plan from symbolic dijkstr's algorithm for multiple DFAs. 
        """
        ts_ycube = reduce(lambda a, b: a & b, self.ts_y_list)
        ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)
        dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)

        prod_accp_state_tuple = self.map_dfa_state_to_tuple(self.target_DFA_list)
        current_ts_dict = {prod_accp_state_tuple: freach_list[max_layer][prod_accp_state_tuple]}
        g_layer = self.manager.addConst(int(max_layer))

        parent_plan = {}

        while not self.check_init_in_bucket(current_ts_dict):
            new_current_ts: dict = {}
            g_layer_plan = {}
            for _to_dfa_state, current_ts in current_ts_dict.items():
                if current_ts.isZero():
                    continue
                sym_current_dfa_states = self.map_dfa_tuple_to_sym_states(_to_dfa_state)
                # compute the valid state predecessors on the DFAs from which we could have transited to the current DFA states given by _to_dfa_states
                valid_pre_dfa_state = self._get_pred_prod_dfa_ts_map(ts_obs_cube=ts_obs_cube,
                                                                     sym_dfa_to_states=sym_current_dfa_states,
                                                                     dfa_ycube=dfa_ycube,
                                                                     verify=False)
                for tr_num, tran_func_action in enumerate(self.ts_transition_fun_list):
                    preds_ts = self.pre_per_action(trans_action=tran_func_action,
                                                   From=current_ts,
                                                   ycube=ts_ycube,
                                                   x_list=self.ts_x_list,
                                                   y_list=self.ts_y_list)
                    
                    if preds_ts.isZero():
                        continue
                    
                    if verbose:
                        self.get_states_from_dd(dd_func=preds_ts,
                                                curr_state_list=self.ts_x_list,
                                                sym_map=self.ts_bdd_sym_to_curr_state_map)
                    
                    # first get the corresponding transition action cost (constant at the terminal node)
                    action_cost_cnst = tran_func_action.findMax()
                    step = g_layer - action_cost_cnst
                    if step.isZero():
                        step_val = 0
                    else:
                        step_val = int(re.findall(r'-?\d+', step.__repr__())[0])
                        # there can be cases where step_val cam go negtive, we skip such iterations
                        if step_val < 0:
                            continue
                    
                    for _from_prod_dfa_state, _v in valid_pre_dfa_state.items():
                        if not _v.isZero():
                            # check the intersection of the corresponding layer's corresponding DFA state with preds_ts_list
                            intersection = self.manager.addZero()
                            if step_val in freach_list:
                                if _from_prod_dfa_state in freach_list[step_val]:
                                    intersection: ADD = freach_list[step_val][_from_prod_dfa_state] & preds_ts
                            else:
                                continue
                        
                            if not intersection.isZero():
                                _valid_intersect: ADD = self.check_from_to_dfa_evolution_validity(from_dfa_state_tuple=_from_prod_dfa_state,
                                                                                                  to_dfa_state_tuple=_to_dfa_state,
                                                                                                  ts_states=intersection,
                                                                                                  ts_xcube=ts_xcube,
                                                                                                  ts_ycube=ts_ycube,
                                                                                                  ts_obs_cube=ts_obs_cube,
                                                                                                  ts_transition_fun=self.ts_transition_fun_list[tr_num])
                                if not _valid_intersect.isZero():
                                    for _ts_state in self.convert_add_cube_to_func(dd_func=_valid_intersect, curr_state_list=self.ts_x_list):
                                        self._append_dict_value(dict_obj=g_layer_plan,
                                                                # key_dfa=f'{_from_prod_dfa_state}->{_to_dfa_states}',
                                                                key_dfa=_from_prod_dfa_state,
                                                                key_ts=self.ts_add_sym_to_curr_state_map[_ts_state],
                                                                value=self.tr_action_idx_map.inv[tr_num])

                                    
                                    if _from_prod_dfa_state in new_current_ts:
                                        new_current_ts[_from_prod_dfa_state] |= _valid_intersect
                                    else:
                                        new_current_ts[_from_prod_dfa_state] = _valid_intersect

                                    parent_plan[step_val] = g_layer_plan

                                    if self.check_init_in_bucket(new_current_ts):
                                        return parent_plan
                                    g_layer = step

            if g_layer.isZero():
                g_int = 0
            else:
                g_int = int(re.findall(r'-?\d+', g_layer.__repr__())[0])
            current_ts_dict = new_current_ts
            
            assert len(current_ts_dict.keys()) != 0, "Looks like something went wrong while retrieving plan. FIX THIS!!!"

            assert  g_int >= 0, "Error Retrieving a plan. FIX THIS!!"

        return parent_plan
