import re 

from functools import reduce
from typing import Union, List


from cudd import Cudd, BDD, ADD

from src.algorithms.base import BaseSymbolicSearch


class SymbolicDijkstraSearch(BaseSymbolicSearch):

    def __init__(self,
                 init_TS: BDD,
                 target_DFA: BDD,
                 init_DFA: BDD,
                 ts_curr_vars: list,
                 ts_next_vars: list,
                 dfa_curr_vars: list,
                 dfa_next_vars: list,
                 ts_obs_vars: list,
                 ts_transition_func: ADD,
                 ts_trans_func_list: List[ADD],
                 dfa_transition_func: BDD,
                 ts_add_sym_to_curr_map: dict,
                 ts_bdd_sym_to_curr_map: dict,
                 ts_bdd_sym_to_S2O_map: dict,
                 ts_add_sym_to_S2O_map: dict,
                 dfa_bdd_sym_to_curr_map: dict,
                 dfa_add_sym_to_curr_map: dict,
                 tr_action_idx_map: dict, 
                 state_obs_add: ADD,
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = init_TS
        self.target_DFA = target_DFA
        self.init_DFA = init_DFA
        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_transition_fun = ts_transition_func
        self.ts_transition_fun_list = ts_trans_func_list
        self.dfa_transition_fun = dfa_transition_func
        self.ts_add_sym_to_curr_state_map: dict = ts_add_sym_to_curr_map
        self.ts_bdd_sym_to_curr_state_map: dict = ts_bdd_sym_to_curr_map
        self.ts_bdd_sym_to_S2obs_map: dict = ts_bdd_sym_to_S2O_map
        self.ts_add_sym_to_S2obs_map: dict = ts_add_sym_to_S2O_map
        self.dfa_bdd_sym_to_curr_state_map: dict = dfa_bdd_sym_to_curr_map
        self.dfa_add_sym_to_curr_state_map: dict = dfa_add_sym_to_curr_map
        self.obs_add = state_obs_add
        self.tr_action_idx_map = tr_action_idx_map
    
    def _append_dict_value(self, dict_obj, key_ts, key_dfa, value):
        """
        Check if key exist in dict or not.

        If Key exist in dict:
           Check if type of value of key is list or not
           
           If type is not list then make it list and Append the value in list
        
        else: add key-value pair
        """
        if key_dfa in dict_obj:
            if key_ts in dict_obj[key_dfa]:
                if not isinstance(dict_obj[key_dfa][key_ts], list):
                    dict_obj[key_dfa][key_ts] = [dict_obj[key_dfa][key_ts]]
                dict_obj[key_dfa][key_ts].append(value)
            else:
                dict_obj[key_dfa][key_ts] = value
        else:
            dict_obj[key_dfa] = {key_ts: value}
    
    def _add_to_rlist(self, bucket: dict, states: ADD):
        """
        A helper function to append the states being added to correcponding
        """
        if isinstance(bucket, None):
            bucket = states
        
        else:
            bucket |= states
    

    def remove_state_explored(self, layer_num: int, open_list: dict, closed: ADD):
        """
        A helper function to remove all the states that have already been explored, i.e., belong to the closed ADD function.
        """
        for _dfa in open_list[layer_num].keys():
            open_list[layer_num][_dfa] = open_list[layer_num][_dfa] & ~closed[_dfa]
    

    def check_open_list_is_empty(self, layer_num: int, open_list: dict):
        """
        A helper function to check if the reached associated with all the dfa states is empty or not
        """
        for _dfa in open_list[layer_num].keys():
            if not open_list[layer_num][_dfa].isZero():
                return False
        
        return True
    
    def check_init_in_bucket(self, bucket: dict):
        """
        A helper function to check if the reached associated with all the dfa states is empty or not
        """
        _dfa_init_state = self.dfa_add_sym_to_curr_state_map[self.init_DFA]
        # for _dfa in bucket.keys():
        if _dfa_init_state in bucket:
            if self.init_TS <= bucket[_dfa_init_state]:
                return True
        
        return False
    

    def get_states_from_dd(self, dd_func: Union[BDD, ADD], curr_state_list: list, sym_map: dict) -> None:
        """
        A function thats wraps arounf convert_cube_to_func() and spits out the states in the corresponding
        """
        #  extract the set of states that are being expanded during each iteration
        if isinstance(dd_func, ADD):
            tmp_dd_func: BDD = dd_func.bddPattern()
            tmp_state_list: List[BDD] = [_avar.bddPattern() for _avar in curr_state_list]
            ts_cube_string = self.convert_cube_to_func(dd_func=tmp_dd_func, curr_state_list=tmp_state_list)
        
        else:
            ts_cube_string = self.convert_cube_to_func(dd_func=dd_func, curr_state_list=curr_state_list)
        
        print("Abstraction State(s) Reached")
        for _s in ts_cube_string:
            _name = sym_map.get(_s)
            assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
            print(_name)
    

    def updated_closed_list(self, closed: ADD, bucket: ADD) -> ADD:
        """
        Update the closed set bu iterating over the entire bucket (all DFA states) reached list,
         and adding them to the expanded set (closed set)
        """
        for _dfa_state, _ts_states in bucket.items():
            closed[_dfa_state] |= _ts_states
        
        return closed
    

    def _get_ts_states_to_dfa_evolution(self, ts_states, dfa_to_state, dfa_from_states):
        """
        A helpr function that give a set of Treansition system system, compute their respective observation.
        Then, given the next DFA state computes the set of predecessors of the DFA state along TS state, that enabled that transtion

        Note: Make sure that the dfa_ to_state variables are in terms of dfa next states and
         dfa_from_states are in terms of dfa_from_state variables.
        """
        dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)
        dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)

        if isinstance(dfa_to_state, ADD):
            dfa_to_state.bddPattern().support() >= dfa_ycube.bddPattern()
        else:
            # the support can contain some or all element of dfa_cubes. Thats is why >= rather than ==
            assert dfa_to_state.support() >= dfa_ycube, "Error while computing TS states for 'to-dfa-state-evolution'. \
            Make sure your 'to-dfa-states' are in terms of next dfa variables "
        
        if isinstance(dfa_from_states, ADD):
            dfa_from_states.bddPattern().support() >= dfa_xcube.bddPattern()
        else:
            assert dfa_from_states.support() >= dfa_xcube, "Error while computing TS states for 'to-dfa-state-evolution'. \
            Make sure your 'from-dfa-states' are in terms of current dfa variables "

        # compute all the possible Transition system states
        ts_cubes = self.convert_add_cube_to_func(dd_func=ts_states, curr_state_list=self.ts_x_list)
        _from_dfa_states_maps = {key: self.manager.addZero() for key in self.dfa_add_sym_to_curr_state_map.inv.keys()}

        for _ts_state in ts_cubes:
            # get their corresponding observation
            _ts_state_obs = self.obs_add.restrict(_ts_state)
            _from_dfa_state = self.dfa_transition_fun.restrict(_ts_state_obs & dfa_to_state)
            # when a state has a True loop (only happens with accepting dfa state), we get multiple _from_dfa_State
            # if len(list(self.convert_add_cube_to_func(dd_func=_from_dfa_state, curr_state_list=self.dfa_x_list))) > 1:
            #     _from_dfa_state = _from_dfa_state & ~dfa_to_state.swapVariables(self.dfa_x_list, self.dfa_y_list)
            
            _from_dfa_cubes = self.convert_add_cube_to_func(dd_func=_from_dfa_state, curr_state_list=self.dfa_x_list)
            for _from_dfa_state in _from_dfa_cubes:
                # assert len(list(_from_dfa_state.generate_cubes())) == 1, "ERROR while computing TS state that correspond to DFA state evolution. \
                #      For a given TS state, we should have at max one transition. FIX THIS!!!"
                _from_dfa_states_maps[self.dfa_add_sym_to_curr_state_map[_from_dfa_state]] |= _ts_state
        
        return _from_dfa_states_maps


    def compute_image_of_composed_graph(self,
                                        open_list: dict,
                                        ts_xcube: ADD,
                                        # dfa_sym_curr_state: ADD,
                                        g_layer: int,
                                        g_val: ADD,
                                        verbose: bool = False):
        """
        From is Open_list[layer_num] which conssits of DFA_state with their corresponding Buckets 

        1. Starting from the init state in the abstraction, we take a step
        2. get the observation for all the states in the image
        3. Check if Any of the state's observation satisfies the DFA edge label 
            3.1 If yes, then transit in the DFA as well.
        """
        _reached_accp_state : bool = False
        # compute the image of the TS states 
        # dfa_curr_state = self.dfa_add_sym_to_curr_state_map[dfa_sym_curr_state]
        
        for tr_action in self.ts_transition_fun_list:
            ts_image_add = self.manager.addZero()
            if _reached_accp_state:
                break
            for _dfa_state in open_list[g_layer]:
                dfa_curr_state = _dfa_state
                dfa_sym_curr_state = self.dfa_add_sym_to_curr_state_map.inv[dfa_curr_state]
                
                # first get the corresponding transition action cost (constant at the terminal node)
                action_cost = tr_action.findMax()
                step = g_val + action_cost
                step_val = int(re.findall(r'\d+', step.__repr__())[0])

                
                image_c = self.image_per_action(trans_action=tr_action,
                                                From=open_list[g_layer][_dfa_state],
                                                xcube=ts_xcube,
                                                x_list=self.ts_x_list,
                                                y_list=self.ts_y_list)
            
                ts_image_add = image_c
                    
                if ts_image_add.isZero():
                    continue
            
                if verbose:
                    self.get_states_from_dd(dd_func=ts_image_add, curr_state_list=self.ts_x_list, sym_map=self.ts_bdd_sym_to_curr_state_map)
                
                # get the observation for all the states in the image
                obs_add = self.obs_add.restrict(ts_image_add)

                # check if any of the DFA edges are satisfied
                image_DFA = self.dfa_transition_fun.restrict(dfa_sym_curr_state & obs_add)
                image_DFA = image_DFA.swapVariables(self.dfa_y_list, self.dfa_x_list)

                if verbose:
                    self.get_states_from_dd(dd_func=image_DFA, curr_state_list=self.dfa_x_list, sym_map=self.dfa_bdd_sym_to_curr_state_map)
            
                if not image_DFA.isZero():
                    # compute which states correspond to which DFA transition 
                    # get each individual cubes from the image of TS
                    # ts_cubes = self.convert_cube_to_func(dd_func=ts_image_add, curr_state_list=self.ts_x_list)
                    ts_cubes = self.convert_add_cube_to_func(dd_func=ts_image_add, curr_state_list=self.ts_x_list)
                    # create a dictionary to store a set of states that that needs to be added to the corresponding DFA fronteir
                    _states_to_be_added_to_reachlist = {key: self.manager.addZero() for key in self.dfa_add_sym_to_curr_state_map.inv.keys()}   
                    for ts_state in ts_cubes:
                        state_obs = self.obs_add.restrict(ts_state)
                        dfa_state = self.dfa_transition_fun.restrict(state_obs & dfa_sym_curr_state)
                        dfa_state = dfa_state.swapVariables(self.dfa_y_list, self.dfa_x_list)
                        if verbose:
                            print(f"State {self.ts_add_sym_to_curr_state_map[ts_state]} caused the DFA to evolve from {dfa_curr_state} to {self.dfa_add_sym_to_curr_state_map[dfa_state]}")
                        _states_to_be_added_to_reachlist[self.dfa_add_sym_to_curr_state_map[dfa_state]] |= ts_state

                    # loop over the tmp dictionary and add them to the their respective reach list
                    for _key, _ts_states in _states_to_be_added_to_reachlist.items():
                        if not _ts_states.isZero():
                            if step_val in open_list:
                                if _key in open_list[step_val]:
                                    open_list[step_val][_key] |= _ts_states
                                else:
                                    open_list[step_val][_key] = _ts_states
                            else:
                                open_list[step_val] = {_key: _ts_states}

                            # early termination criteria
                            if _key == self.dfa_add_sym_to_curr_state_map[self.target_DFA]:
                                print("**************Found a Plan, Now retrieving it!**************")
                                _reached_accp_state: bool = True
                                break


    def symbolic_dijkstra(self, verbose: bool = False):
        """
        Implement a general action cost planning algorithm using Dijkstra's algorithm in a symbolic fashion.

        To represent the cost along with an action, we use ADD instead of BDDs.

        NOTE: If we were to represent edge cost as boolean variables with max cost of C_max, then we would have
         required Upper bounds(log2(C_max + 1)) boolean variables. So, for a tiny graph with very large wieghts this
         would scale poorly. Instead we use ADDs to mitigate this problem as we only need as many constant variables
         as the numbers of actions costs in the causal graph (graph over which we are planning).
        """

        # TODO: In future change thisn  - This is dictionary that stores our bucksets as list. BUt we use dictionary becuase,
        # 1. It is faster than list 2. You do not have to create placeholders for buckets before hand
        open_list = {}

        ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)

        closed = {key: self.manager.addZero() for key in self.dfa_add_sym_to_curr_state_map.inv.keys()}
        g_val = self.manager.addZero()
        if g_val.isZero():
            g_layer = 0

        dfa_curr_state = self.init_DFA
        dfa_final_state = self.dfa_add_sym_to_curr_state_map[self.target_DFA]
        
        open_list[g_layer] = {self.dfa_add_sym_to_curr_state_map[dfa_curr_state]: self.init_TS}

        while not dfa_final_state in open_list[g_layer].keys():
            # remove all states that have been explored
            self.remove_state_explored(layer_num=g_layer, open_list=open_list, closed=closed)

            # If unexpanded states with total cost g exist ... 
            if not self.check_open_list_is_empty(layer_num=g_layer, open_list=open_list):
                # Add states to be expanded next to already expanded states
                closed = self.updated_closed_list(closed, open_list[g_layer])
                # Find successors and add them to their respectice buckets
                self.compute_image_of_composed_graph(open_list=open_list,
                                                     g_layer=g_layer,
                                                     g_val=g_val,
                                                     ts_xcube=ts_xcube,
                                                    #  dfa_sym_curr_state=dfa_curr_state,
                                                     verbose=verbose)

            g_val = g_val + self.manager.addOne()
            g_layer += 1
        
        # open_list[g_layer][dfa_final_state] = open_list[g_layer][dfa_final_state] & self.target_DFA

        return self.retrieve_dijkstra(max_layer=g_layer, add_freach_list=open_list, verbose=verbose)
    
    def retrieve_dijkstra(self, max_layer: int, add_freach_list: dict, verbose: bool = False):
        """
        Retrieval of a plan for symbolic Dijkstra
        """
        # plan = []
        # current = add_freach_list[max_layer]
        ts_ycube = reduce(lambda a, b: a & b, self.ts_y_list)
        g_layer = self.manager.addConst(int(max_layer))
        dfa_final_state = self.dfa_add_sym_to_curr_state_map[self.target_DFA]
        current_ts_dict = {dfa_final_state: add_freach_list[max_layer][dfa_final_state]}
        ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)
        dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)
        # g_val = int(re.findall(r'\d+', g_layer.__repr__())[0])
        # current_dfa = 'accept_all'
        print("Working Retrieval plan now")
        

        parent_plan = {}

        while not self.check_init_in_bucket(current_ts_dict):
            # preds_ts_list = []
            # valid_pred_ts = self.manager.addZero()
            new_current_ts: dict = {key: self.manager.addZero() for key in self.dfa_add_sym_to_curr_state_map.inv.keys()}
            for tr_num, tran_func_action in enumerate(self.ts_transition_fun_list):
                for _to_dfa_state, current_ts in current_ts_dict.items():
                    if current_ts.isZero():
                        continue
                    sym_current_dfa = self.dfa_add_sym_to_curr_state_map.inv[_to_dfa_state]
                    preds_ts = self.pre_per_action(trans_action=tran_func_action,
                                                   From=current_ts,
                                                   ycube=ts_ycube,
                                                   x_list=self.ts_x_list,
                                                   y_list=self.ts_y_list)
                    if preds_ts.isZero():
                        continue
                    if verbose:
                        self.get_states_from_dd(dd_func=preds_ts, curr_state_list=self.ts_x_list, sym_map=self.ts_bdd_sym_to_curr_state_map)

                    # first get the corresponding transition action cost (constant at the terminal node)
                    action_cost_cnst = tran_func_action.findMax()
                    step = g_layer - action_cost_cnst
                    if step.isZero():
                        step_val = 0
                    else:
                        step_val = int(re.findall(r'\d+', step.__repr__())[0])

                
                    # compute the pre on the DFA
                    # check any of the above pre intersects with their respective forward set
                    dfa_w_no_obs = self.dfa_transition_fun.existAbstract(ts_obs_cube)

                    pred_dfa = self.pre_per_action(trans_action=dfa_w_no_obs,
                                                   From=sym_current_dfa,
                                                   ycube=dfa_ycube,
                                                   x_list=self.dfa_x_list,
                                                   y_list=self.dfa_y_list)    # this is in terms of current dfa state variables
                    
                    valid_pre_dfa_state = self._get_ts_states_to_dfa_evolution(ts_states=current_ts,
                                                                               dfa_to_state=sym_current_dfa.swapVariables(self.dfa_x_list, self.dfa_y_list),
                                                                               dfa_from_states=pred_dfa)
                    
                    # compute the intersetion of
                    for _from_dfa_state, _v in valid_pre_dfa_state.items():
                        if not _v.isZero():

                            # check the intersection of the corresponding layer's corresponding DFA state with preds_ts_list
                            try:
                                intersection: ADD = add_freach_list[step_val][_from_dfa_state] & preds_ts
                            except:
                                if verbose:
                                    print(f"The Forward reach list dose not contain {_from_dfa_state} in bucket {step_val}.\
                                        For now skipping computing this intersection. This may cause issues in the future.")
                                continue

                            if verbose and not intersection.isZero():
                                self.get_states_from_dd(dd_func=intersection, curr_state_list=self.ts_x_list, sym_map=self.ts_bdd_sym_to_curr_state_map)
                            
                            if not intersection.isZero():
                                # valid_pred_ts |= intersection
                                for _ts_state in self.convert_add_cube_to_func(dd_func=intersection, curr_state_list=self.ts_x_list):
                                    self._append_dict_value(dict_obj=parent_plan,
                                                            key_dfa=_from_dfa_state,
                                                            key_ts=self.ts_add_sym_to_curr_state_map[_ts_state],
                                                            value=self.tr_action_idx_map.inv[tr_num])
                    
                                # this is our current_ts now
                                # assert not valid_pred_ts.isZero(), "Error retireving a plan. The intersection of Forward and Backwards Reachable sets should NEVER be empty. FIX THIS!!!"
                                new_current_ts[_from_dfa_state] |= intersection


            current_ts_dict = new_current_ts
            # sym_current_dfa = self.dfa_add_sym_to_curr_state_map.inv[_dfa_state]
            g_layer = step
        
        return parent_plan











