'''
This file implements Symbolic Graph search algorithms
'''
import sys
import copy

from bidict import bidict
from functools import reduce
from cudd import Cudd, BDD, ADD

from typing import List
from itertools import product


class SymbolicSearch(object):
    """
    Given a Graph, find the shortest path as per the symbolic A* (BDDA*) algorithm as outlined by Jensen, Bryant, Valeso's paper.
    """

    def __init__(self,
                 init,
                 init_TS,
                 target_DFA,
                 init_DFA,
                 target,
                 manager,
                 ts_curr_vars: list,
                 ts_next_vars: list,
                 ts_obs_vars: list,
                 dfa_curr_vars: list,
                 dfa_next_vars: list,
                 ts_transition_func,
                 ts_trans_func_list,
                 dfa_transition_func,
                 ts_sym_to_curr_map: dict,
                 ts_sym_to_S2O_map: dict,
                 dfa_sym_to_curr_map: dict,
                 tr_action_idx_map: dict, 
                 state_obs_bdd):

        self.init = init
        self.init_TS = init_TS
        self.target_DFA = target_DFA
        self.init_DFA = init_DFA
        self.target = target
        self.manager = manager
        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_obs_list = ts_obs_vars
        self.ts_transition_fun = ts_transition_func
        self.ts_transition_fun_list = ts_trans_func_list
        self.dfa_transition_fun = dfa_transition_func
        self.ts_sym_to_curr_state_map: dict = ts_sym_to_curr_map
        self.ts_sym_to_S2obs_map: dict = ts_sym_to_S2O_map
        self.dfa_sym_to_curr_state_map: dict = dfa_sym_to_curr_map
        self.obs_bdd = state_obs_bdd
        self.tr_action_idx_map = tr_action_idx_map
        # self.transition_fun = self._build_transition_function_w_action_costs()
        # self.estimate_fun = self._build_estimate_function()
        # self.reached = []
        # self.que = []
    
    def pre(self, From, ycube, x_list, y_list, transition_fun):
        """
        Compute the predecessors of 'From'.
        
        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        fromY = From.swapVariables(x_list, y_list)
        return transition_fun.andAbstract(fromY, ycube)
    
    def pre_per_action(self, trans_action, From, ycube, x_list, y_list):
        """
         Compute the predecessors of 'From' under action specific transition function.

        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        fromY = From.swapVariables(x_list, y_list)
        if type(fromY) == type(self.manager.addZero()):
            _conjoin = trans_action.bddPattern() & fromY.bddPattern()
            return _conjoin.existAbstract(ycube.bddPattern()).toADD()
        else:
            return trans_action.andAbstract(fromY, ycube)
    

    def image(self, From, xcube, x_list, y_list, transition_fun):
        """
        Compute the set of possible state reachable from 'From' state.

        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        # check if its add or bdd
        if type(From) == type(self.manager.addZero()):
            _conjoin = transition_fun & From
            ImgY = _conjoin.existAbstract(xcube)
        else:
            ImgY = transition_fun.andAbstract(From, xcube)

        return ImgY.swapVariables(y_list, x_list)
    
    
    def image_per_action(self, trans_action, From, xcube, x_list, y_list):
        """
        Compute the set of possible state reachable from 'From' state under action specific transition function.

        andAbstract: Conjoin to another BDD and existentially quantify variables.

        If the Structures are ADD, we then convet them to BDD using bddPattern() method. We then compute the image
         (of type BDD) and then convert to add using toADD() and return the variables.
        """
        if type(From) == type(self.manager.addZero()):
            _conjoin = trans_action.bddPattern() & From.bddPattern()
            # _conjoin = trans_action & From
            ImgY = _conjoin.existAbstract(xcube.bddPattern())

            return ImgY.swapVariables(y_list, x_list).toADD()
        else:
            ImgY = trans_action.andAbstract(From, xcube)

            return ImgY.swapVariables(y_list, x_list)
    

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
    

    def _get_ts_states_to_dfa_evolution(self, ts_states, dfa_to_state, dfa_from_states):
        """
        A helpr function that give a set of Treansition system system, compute their respective observation.
        Then, given the next DFA state computes the set of predecessors of the DFA state along TS state, that enabled that transtion

        Note: Make sure that the dfa_ to_state variables are in terms of dfa next states and
         dfa_from_states are in terms of dfa_from_state variables.
        """
        dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)
        dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)
        # the support can contain some or all element of dfa_cubes. Thats is why >= rather than ==
        assert dfa_to_state.support() >= dfa_ycube, "Error while computing TS states for 'to-dfa-state-evolution'. \
         Make sure your 'to-dfa-states' are in terms of next dfa variables "
        
        assert dfa_from_states.support() >= dfa_xcube, "Error while computing TS states for 'to-dfa-state-evolution'. \
         Make sure your 'from-dfa-states' are in terms of current dfa variables "

        # compute all the possible Transition system states
        ts_cubes = self._convert_cube_to_func(bdd_func=ts_states, curr_state_list=self.ts_x_list)
        _from_dfa_states_maps = {key: self.manager.bddZero() for key in self.dfa_sym_to_curr_state_map.inv.keys()}

        for _ts_state in ts_cubes:
            # get their corresponding observation
            _ts_state_obs = self.obs_bdd.restrict(_ts_state)
            _from_dfa_state = self.dfa_transition_fun.restrict(_ts_state_obs & dfa_to_state)
            # when a state has a True loop (only happens with accepting dfa state), we get multiple _from_dfa_State
            # _from_dfa_state = _from_dfa_state & ~self.dfa_sym_to_curr_state_map.inv['accept_all']
            if len(list(self._convert_cube_to_func(bdd_func=_from_dfa_state, curr_state_list=self.dfa_x_list))) > 1:
                _from_dfa_state = _from_dfa_state & ~dfa_to_state.swapVariables(self.dfa_x_list, self.dfa_y_list)
            
            assert len(list(_from_dfa_state.generate_cubes())) == 1, "ERROR while computing TS state thar correspond to DFA state evolution. \
                 For a given TS state, we should have at max one transition. FIX THIS!!!"
            _from_dfa_states_maps[self.dfa_sym_to_curr_state_map[_from_dfa_state]] |= _ts_state
        
        return _from_dfa_states_maps
    

    def _convert_cube_to_func_S2Obs(self, bdd_func: str) -> List[BDD]:
        """
        A helper function that convert a string of a path to its corresponding boolean form for a Given transition system's state observatopn
        """

        # NOTE: Add a check where we skip vairables that are irrelevant to the state. 
        if isinstance(bdd_func, ADD):
            bdd_func = bdd_func.bddPattern()

        bddVars = []
        for cube in bdd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                # if var == 2 and self.manager.bddVar(_idx) not in self.ts_obs_list: 
                #     continue   # skipping over prime states 
                if self.manager.bddVar(_idx) in self.ts_obs_list:
                    if var == 2:
                        _amb_var.append([self.manager.bddVar(_idx), ~self.manager.bddVar(_idx)])   # count how many vars are missing to fully define the bdd
                    if var == 0 :
                        var_list.append(~self.manager.bddVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.bddVar(_idx))

            # check if it is not full defined
            # og_var_ls = copy.deepcopy(var_list)
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    bddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                bddVars.append(reduce(lambda a, b: a & b, var_list))
        return bddVars


    def _convert_cube_to_func(self, bdd_func: str, curr_state_list: list) -> List[BDD]:
        """
        A helper function that convert a string of a path to its corresponding boolean form for a Given transition system
        """

        # NOTE: Add a check where we skip vairables that are irrelevant to the state. 
        if isinstance(bdd_func, ADD):
            bdd_func = bdd_func.bddPattern()

        bddVars = []
        for cube in bdd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if var == 2 and self.manager.bddVar(_idx) not in curr_state_list:   # not x list is better than y _list because we also have dfa vairables 
                    continue   # skipping over prime states 
                
                elif self.manager.bddVar(_idx) in curr_state_list:
                    if var == 2:
                        _amb_var.append([self.manager.bddVar(_idx), ~self.manager.bddVar(_idx)])   # count how many vars are missing to fully define the bdd
                    elif var == 0:
                        var_list.append(~self.manager.bddVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.bddVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                        sys.exit(-1)

            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    bddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                bddVars.append(reduce(lambda a, b: a & b, var_list))
        
        # make sure that every element in bddVars container all the bdd variables needed to define it completely
        for ele in bddVars:
            # conver ele to stirng and the bddVars to str and check if all of them exisit or not!
            _str_ele = ele.__repr__()
            for bVar in curr_state_list:
                assert str(bVar) in _str_ele, "Error! The cube does not contain all the boolean variables in it! FIX THIS!!"

        return bddVars


    def symbolic_bfs(self, verbose: bool = False):
        """
        Implement a symbolic bread first search algorithm.
        """
        reached_list = []
        reached = self.init
        closed = self.manager.bddZero()
        layer = 0
        xcube = reduce(lambda x, y: x & y, self.x_list)

        reached_list.append(reached)
        while not self.target <= reached_list[layer]:

            reached_list[layer] = reached_list[layer] & ~closed

            if reached_list[layer] == self.manager.bddZero():
                print("No plan found")
                break

            closed |= reached_list[layer]
            
            # for franka world - if reached set consists only of the init set then next set of states is the
            # INTERSECTION of the image of each init state

            # if reached_list[layer].compare(self.init, 2):
            #     # first compute the individual initial states 
            #     init_states = self._convert_cube_to_func(bdd_func=reached_list[layer])
            #     # take the intersection of the states 
            #     next_state = self.manager.bddOne()
            #     for _s in init_states:
            #         print(self.ts_sym_to_state_curr_map.get(_s))
            #         test = self._convert_cube_to_func(self.image(From=_s, xcube=xcube))
            #         next_state = next_state & self.image(From=_s, xcube=xcube)
            #         for t in test:
            #             print(self.ts_sym_to_state_curr_map.get(t))

            #     reached_list.append(next_state)
            
            # else:
            # reached_list.append(self.image(From=reached_list[layer], xcube=xcube))

            image_bdd = self.manager.bddZero()
            for tr_action in self.transition_fun_list:
                image_c = self.image_per_action(trans_action=tr_action,
                                                From=reached_list[layer],
                                                xcube=xcube,
                                                x_list=self.ts_x_list,
                                                y_list=self.ts_y_list)
                image_bdd |= image_c

            reached_list.append(image_bdd)

            
            if verbose:
                # now extract the set of states that are being expanded during each iteration
                test = self._convert_cube_to_func(bdd_func=reached_list[layer])
                for _s in test:
                    _name = self.ts_sym_to_state_curr_map.get(_s)
                    if _name is None:
                        print('Hi!')
                        sys.exit(-1)
                    print(self.ts_sym_to_state_curr_map.get(_s))
                    

            layer += 1

        reached_list[layer] = reached_list[layer] & self.target

        # return self.retrieve_bfs(reached_list)
        return self.retrive_bfs_action(reached_list)
    

    def _check_target_dfa_in_closed_list(self, reached_list) -> bool:
        """
        A helper function to check if the DFA target state has been reached or not
        """
        for _, sub_reached_list in reached_list.items():
            if len(sub_reached_list['reached_list']) > 0 and self.target_DFA <= sub_reached_list['reached_list'][-1]:
                return True
        return False



    def symbolic_bfs_wLTL(self, max_ts_state: int, verbose: bool = False):
        """
        Implement a symbolic bread first search algorithm for LTL based planning.

        Reach: The set of all states
        Closed: 

        1. Create a List of list. Each list corresponds the fronteir we are expanding for each DFA state. Then 

        In this search algorithm, we start from the init state in both the TS and DFA.
        1. Starting from the init state in the abstraction, we take a step
        2. get the observation for all the states in the image
        3. Check if Any of the state's observation satisfies the DFA edge label 
            3.1 If yes, then transit in the DFA as well.
            3.2 Repeat the above process, until we reach the accepting state in the DFA
        4. If a valid path exist, retrieve it.
        """
        # get the # number of states in the TS 
        parent_reached_list = {key : {'reached_list' : [self.manager.bddZero() for _ in range(4*max_ts_state)],
                                      'dfa_counter' : 0,
                                      'closed': self.manager.bddZero()} for key in self.dfa_sym_to_curr_state_map.inv.keys()}
        # maintain a common layering number
        parent_layer_counter = 0
        ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)

        # add the initi TS state to the init DFA state
        _explicit_dfa_state: str = self.dfa_sym_to_curr_state_map[self.init_DFA]
        parent_reached_list[_explicit_dfa_state]['reached_list'][parent_layer_counter] = self.init_TS

        
        while parent_reached_list["accept_all"]['reached_list'][parent_layer_counter].isZero():
            for _dfa_curr_state, _dfa_fronteirs in parent_reached_list.items():
                if parent_layer_counter > _dfa_fronteirs['dfa_counter']:
                    _local_layer_counter = _dfa_fronteirs['dfa_counter']
                    reached = _dfa_fronteirs['reached_list'][_local_layer_counter]
                elif parent_layer_counter == _dfa_fronteirs['dfa_counter']:
                    _local_layer_counter = parent_layer_counter
                    reached = _dfa_fronteirs['reached_list'][parent_layer_counter]
                else:
                    print("Error with counters in BFS algorithm. FIX THIS!!!")
                    sys.exit(-1)
                closed = _dfa_fronteirs['closed']
                if not reached.isZero():
                    if (parent_reached_list[_dfa_curr_state]['reached_list'][_local_layer_counter] & ~closed ).isZero():
                        print(f"**************Reached a Fixed Point for DFA State {_dfa_curr_state}**************")
                        continue

                    parent_reached_list[_dfa_curr_state]['reached_list'][_local_layer_counter] = \
                     parent_reached_list[_dfa_curr_state]['reached_list'][_local_layer_counter] & ~closed
                    
                    parent_reached_list[_dfa_curr_state]['closed'] |= \
                                 parent_reached_list[_dfa_curr_state]['reached_list'][_local_layer_counter]
                    
                    # if there is a state in the TA that reached the DFA accepting state then terminate
                    if not parent_reached_list["accept_all"]['reached_list'][_local_layer_counter].isZero():
                        print("************** Reached an accepting state. Now Retrieving a plan **************")
                        break 

                    # compute the image of the TS states 
                    ts_image_bdd = self.manager.bddZero()
                    for tr_action in self.ts_transition_fun_list:
                        image_c = self.image_per_action(trans_action=tr_action,
                                                        From=parent_reached_list[_dfa_curr_state]['reached_list'][_local_layer_counter],
                                                        xcube=ts_xcube,
                                                        x_list=self.ts_x_list,
                                                        y_list=self.ts_y_list)
                        ts_image_bdd |= image_c
                    
                    if verbose:
                        # now extract the set of states that are being expanded during each iteration
                        ts_cube_string = self._convert_cube_to_func(bdd_func=ts_image_bdd, curr_state_list=self.ts_x_list)
                        print("Abstraction State(s) Reached")
                        for _s in ts_cube_string:
                            _name = self.ts_sym_to_curr_state_map.get(_s)
                            assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                            print(_name)

                    
                    # get the observation for all the states in the image
                    obs_bdd = self.obs_bdd.restrict(ts_image_bdd)

                    # check if any of the DFA edges are satisfied
                    _dfa_sym_state = self.dfa_sym_to_curr_state_map.inv[_dfa_curr_state]
                    image_DFA = self.dfa_transition_fun.restrict(_dfa_sym_state & obs_bdd)
                    image_DFA = image_DFA.swapVariables(self.dfa_y_list, self.dfa_x_list)

                    if verbose:
                        dfa_cube_string = self._convert_cube_to_func(bdd_func=image_DFA, curr_state_list=self.dfa_x_list)
                        print("DFA State(s) Reached")
                        for _s in dfa_cube_string:
                            _name = self.dfa_sym_to_curr_state_map.get(_s)
                            assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                            print(_name)

                    if not image_DFA.isZero():
                        # compute which states correspond to which DFA transition 
                        # get each individual cubes from the image of TS
                        ts_cubes = self._convert_cube_to_func(bdd_func=ts_image_bdd, curr_state_list=self.ts_x_list)
                        # create a dictionary to store a set of states that that needs to be added to the corresponding DFA fronteir
                        _states_to_be_added_to_reachlist = {key: self.manager.bddZero() for key in self.dfa_sym_to_curr_state_map.inv.keys()}   
                        for ts_state in ts_cubes:
                            state_obs = self.obs_bdd.restrict(ts_state)
                            dfa_state = self.dfa_transition_fun.restrict(state_obs & _dfa_sym_state)
                            dfa_state = dfa_state.swapVariables(self.dfa_y_list, self.dfa_x_list)
                            if verbose:
                                print(f"State {self.ts_sym_to_curr_state_map[ts_state]} caused the DFA to evolve from {_dfa_curr_state} to {self.dfa_sym_to_curr_state_map[dfa_state]}")
                            _states_to_be_added_to_reachlist[self.dfa_sym_to_curr_state_map[dfa_state]] |= ts_state 
                        
                        # loop over the tmp dictionary and add them to the their respective reach list
                        for _key, _ts_states in _states_to_be_added_to_reachlist.items():
                            if not _ts_states.isZero():
                                # check if the state begin added has a;ready been explored or not for this DFA state
                                _new_ts_state = _ts_states & ~parent_reached_list[_key]['closed']
                                if not _new_ts_state.isZero():
                                    # if you are adding to the current then append, else add to the last layer of the other boxes
                                    # if _key == _dfa_curr_state:
                                    parent_reached_list[_key]['reached_list'][parent_layer_counter + 1] = _new_ts_state
                                    parent_reached_list[_key]['dfa_counter'] = parent_layer_counter + 1
                   
                    # update the layer after you have looped through all the DFA states
                    parent_layer_counter += 1
                
        
        # sanity check printing
        if not parent_reached_list["accept_all"]['reached_list'][parent_layer_counter].isZero():
            print("************** Reached an accepting state. Now Retrieving a plan**************")
            return self.retrive_bfs_wLTL_actions(reached_list_composed=parent_reached_list,
                                                        max_layer_num=parent_layer_counter,
                                                        verbose=verbose)
        else:
            print("No plan found")
            sys.exit()


    def retrieve_bfs(self, reached_list):
        """
        Retrieve the plan from symbolic BFS algorithm
        """
        ycube = reduce(lambda a, b: a & b, self.y_list)
        plan = []
        n = len(reached_list)
        current = reached_list[n-1]
        plan.append(current)

        for layer in reversed(range(n)):
            pred = self.pre(From=current, ycube=ycube)
            if pred & reached_list[layer - 1] != self.manager.bddZero():
                current = pred & reached_list[layer - 1]
                plan.append(current)

        print(plan)
    
    def retrive_bfs_action(self, reached_list):
        """
        Retrieve the plan from symbolic BFS algorithm
        """
        ycube = reduce(lambda a, b: a & b, self.y_list)
        plan = []
        n = len(reached_list)
        current = reached_list[n - 1]

        for layer in reversed(range(n)):
            # for each action
            for idx, tran_func_action in enumerate(self.transition_fun_list):
                pred = self.pre_per_action(trans_action=tran_func_action, From=current, ycube=ycube)
                if pred & reached_list[layer - 1] != self.manager.bddZero():
                    current = pred & reached_list[layer - 1]
                    plan.append(idx)

        print(plan)

        return plan
    

    def retrive_bfs_wLTL_actions(self, reached_list_composed, max_layer_num: int, verbose: bool = False):
        """
        Retrieve the plan from symbolic BFS algorithm. The list is sequence of composed states of TS and DFA.
        """
        ts_ycube = reduce(lambda a, b: a & b, self.ts_y_list)
        accepting_dfa_count = reached_list_composed['accept_all']['dfa_counter']
        current_ts = reached_list_composed['accept_all']['reached_list'][accepting_dfa_count]
        current_dfa = 'accept_all'
        sym_current_dfa = self.dfa_sym_to_curr_state_map.inv[current_dfa]
        dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)
        ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)
        reverse_count = 1    # counter to keep track of No. of backsteps we take. Helps when we are jumming from DFA state to another 

        # loops till you reach the init DFA and TS state
        parent_plan = {}
        iter_count = 0
        while not(self.init_TS <= current_ts and sym_current_dfa == self.init_DFA):
            # for each action
            preds_ts_list = []
            for tran_func_action in self.ts_transition_fun_list:
                preds_ts_list.append(self.pre_per_action(trans_action=tran_func_action,
                                                         From=current_ts,
                                                         ycube=ts_ycube,
                                                         x_list=self.ts_x_list,
                                                         y_list=self.ts_y_list))
               
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

            # check if any of the values are not zero
            for _dfa_state, _v in valid_pre_dfa_state.items():
                if not _v.isZero():
                    # we ignore the very last layer as that is the list where the current ts state came from 
                    # reached_list_composed[_dfa_state]['layer'] = reverse_count
                    _layer_num = max_layer_num - reverse_count
                    while reached_list_composed[_dfa_state]['reached_list'][_layer_num].isZero():
                        reverse_count += 1
                        _layer_num = max_layer_num - reverse_count

                    valid_pred_ts = self.manager.bddZero() 
                    for tr_num, pred_ts_bdd in enumerate(preds_ts_list):
                        _valid_ts = pred_ts_bdd & reached_list_composed[_dfa_state]['reached_list'][_layer_num] 
                        if verbose:
                            # print the set of states that are lie at the intersection of Backward Search and Forward Search
                            ts_cube_string = self._convert_cube_to_func(bdd_func=_valid_ts, curr_state_list=self.ts_x_list)
                            # print("Abstraction State(s) Reached")
                            for _s in ts_cube_string:
                                _name = self.ts_sym_to_curr_state_map.get(_s)
                                assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                                print(f"{_name} lies at the intersection of Forward and Backward Search at Iter {iter_count}")
                        # tr_num corresponds to the TR action
                        if not _valid_ts.isZero():
                            valid_pred_ts |= _valid_ts
                            for _ts_state in self._convert_cube_to_func(bdd_func=_valid_ts, curr_state_list=self.ts_x_list):
                                self._append_dict_value(dict_obj=parent_plan,
                                                        key_dfa=_dfa_state,
                                                        key_ts=self.ts_sym_to_curr_state_map[_ts_state],
                                                        value=self.tr_action_idx_map.inv[tr_num])
                    # this is our current_ts now
                    assert not valid_pred_ts.isZero(), "Error retireving a plan. The intersection of Forward anc Backwards Reachable sets should NEVER be empty. FIX THIS!!!"
                    current_ts = valid_pred_ts
                    sym_current_dfa = self.dfa_sym_to_curr_state_map.inv[_dfa_state]
            # normal counter for iteration count
            iter_count += 1
            reverse_count += 1

        return parent_plan