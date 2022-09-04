'''
This file implements Symbolic Graph search algorithms
'''
import sys
import copy

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
                 ts_curr_vars,
                 ts_next_vars,
                 ts_obs_vars,
                 dfa_curr_vars,
                 dfa_next_vars,
                 ts_transition_func,
                 ts_trans_func_list,
                 dfa_transition_func,
                 ts_sym_to_curr_map,
                 ts_sym_to_S2O_map,
                 dfa_sym_to_curr_map,
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
        # self.transition_fun = self._build_transition_function_w_action_costs()
        # self.estimate_fun = self._build_estimate_function()
        # self.reached = []
        # self.que = []
    
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
                    var_list.append(_ele[0])
                    bddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list.pop()
            else:
                bddVars.append(reduce(lambda a, b: a & b, var_list))
            
            # # before returning assert that the length of bdd Vars is same cube length
            # assert len(bddVars[-1]) == len(curr_state_list), "Error Converting cube to boolean string"

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



    def updated_symbolic_bfs_wLTL(self, verbose: bool = False):
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
        parent_reached_list = {key : {'reached_list' : [],
                                      'layer': 0,
                                      'closed': self.manager.bddZero()} for key in self.dfa_sym_to_curr_state_map.inv.keys()}

        ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)

        # add the initi TS state to the init DFA state
        _explicit_dfa_state: str = self.dfa_sym_to_curr_state_map[self.init_DFA]
        parent_reached_list[_explicit_dfa_state]['reached_list'].append(self.init_TS)

        
        while not len(parent_reached_list["accept_all"]['reached_list']) > 0:
            for _dfa_curr_state, _dfa_fronteirs in parent_reached_list.items():
                layer_num = _dfa_fronteirs['layer'] 
                if len(_dfa_fronteirs['reached_list']) > 0:
                    reached = _dfa_fronteirs['reached_list'][layer_num]
                else:
                    reached = []
                closed = _dfa_fronteirs['closed']
                if len(_dfa_fronteirs['reached_list']) > 0 and not reached.isZero():
                    parent_reached_list[_dfa_curr_state]['reached_list'][layer_num] = \
                     parent_reached_list[_dfa_curr_state]['reached_list'][layer_num] & ~closed
                    
                    parent_reached_list[_dfa_curr_state]['closed'] |= \
                                 parent_reached_list[_dfa_curr_state]['reached_list'][layer_num]
                    
                    if parent_reached_list[_dfa_curr_state]['reached_list'][layer_num].isZero():
                        print(f"**************Reached a Fixed Point for DFA State {_dfa_curr_state}**************")

                    # if there is a state in the TA that reached the DFA accepting state then terminate
                    if len(parent_reached_list["accept_all"]['reached_list']) > 0:
                        break 

                    # compute the image of the TS states 
                    ts_image_bdd = self.manager.bddZero()
                    for tr_action in self.ts_transition_fun_list:
                        image_c = self.image_per_action(trans_action=tr_action,
                                                        From=parent_reached_list[_dfa_curr_state]['reached_list'][layer_num],
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
                                    parent_reached_list[_key]['reached_list'].append(_new_ts_state)
            
                    # update the layer after you have looped through all the DFA states
                    _dfa_fronteirs['layer'] += 1
                    if len(_dfa_fronteirs['reached_list']) -1 != _dfa_fronteirs['layer']:
                        _dfa_fronteirs['layer'] -= 1
        
        return 

        
    
    def symbolic_bfs_wLTL(self, verbose: bool = False):
        """
        Implement a symbolic bread first search algorithm for LTL based planning.

        In this search algorithm, we start from the init state in both the TS and DFA.
        1. Starting from the init state in the abstraction, we take a step
        2. get the observation for all the states in the image
        3. Check if Any of the state's observation satisfies the DFA edge label 
            3.1 If yes, then transit in the DFA as well.
            3.2 Repeat the above process, until we reach the accepting state in the DFA
        4. If a valid path exist, retrieve it.
        """
        reached_list_composed: list = []
        closed_composed = self.manager.bddZero()

        reached_list_TS: list = []
        reached_TS = self.init_TS
        closed_TS = self.manager.bddZero()

        layer_TS = 0
        ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)

        reached_list_DFA: list = []
        reached_DFA = self.init_DFA
        closed_DFA = self.manager.bddZero()
        layer_DFA = 0
        
        dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)

        reached_list_TS.append(reached_TS)
        reached_list_DFA.append(reached_DFA)
        reached_list_composed.append(reached_TS & reached_DFA)
        while not self.target_DFA <= reached_list_DFA[layer_DFA]:

            reached_list_TS[layer_TS] = reached_list_TS[layer_TS] & ~closed_TS
            reached_list_DFA[layer_DFA] = reached_list_DFA[layer_DFA] & ~closed_DFA
            reached_list_composed[layer_TS] = reached_list_composed[layer_TS]  & ~closed_composed

            if reached_list_TS[layer_TS] == self.manager.bddZero():
                print("The specifcation is unrealizable")
                break

            closed_TS |= reached_list_TS[layer_TS]
            closed_DFA |= reached_list_DFA[layer_DFA]
            closed_composed |= reached_list_composed[layer_TS]

            # evolve on the Transition system
            ts_image_bdd = self.manager.bddZero()
            for tr_action in self.ts_transition_fun_list:
                image_c = self.image_per_action(trans_action=tr_action,
                                                From=reached_list_TS[layer_TS],
                                                xcube=ts_xcube,
                                                x_list=self.ts_x_list,
                                                y_list=self.ts_y_list)
                ts_image_bdd |= image_c

            # get the observation for all the states in the image
            obs_bdd = self.obs_bdd.restrict(ts_image_bdd)
            reached_list_TS.append(ts_image_bdd)

            if verbose:
                # now extract the set of states that are being expanded during each iteration
                ts_cube_string = self._convert_cube_to_func(bdd_func=ts_image_bdd, curr_state_list=self.ts_x_list)
                # ts_cube_string = self._convert_cube_to_func(bdd_func=reached_list[-1], curr_state_list=self.ts_x_list)
                print("Abstraction State(s) Reached")
                for _s in ts_cube_string:
                    _name = self.ts_sym_to_curr_state_map.get(_s)
                    assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                    print(_name)
                
                # this is for checking lables associated with state we expanded to
                ts_S2O_cube_String = self._convert_cube_to_func_S2Obs(bdd_func=obs_bdd)
                print("Corresponding State Observation")
                for _s in ts_S2O_cube_String:
                    _name = self.ts_sym_to_S2obs_map.get(_s)
                    assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                    print(_name)

            # check if any of the DFA edges are satisfied
            image_DFA = self.dfa_transition_fun.restrict(reached_list_DFA[-1] & obs_bdd)

             # you get a characteristic function which is a combination of ts_obs_labels and dfa_xcube.
            #  Intuitively, this means, based on which state you are in, your label will differ and accordingly you may or may not evolve on the DFA.
            # image_DFA = image_DFA.existAbstract(ts_obs_cube)   # we needs to swap vairables
            image_DFA = image_DFA.swapVariables(self.dfa_y_list, self.dfa_x_list)

            image_DFA_only = image_DFA.existAbstract(ts_obs_cube)
            reached_list_DFA.append(image_DFA_only)   # using exist abstract removes the dependence on TA curr state related boolean variables 
            reached_list_composed.append(image_DFA_only & ts_image_bdd)

            if verbose:
                dfa_cube_string = self._convert_cube_to_func(bdd_func=image_DFA, curr_state_list=self.dfa_x_list)
                print("DFA State(s) Reached")
                for _s in dfa_cube_string:
                    _name = self.dfa_sym_to_curr_state_map.get(_s)
                    assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                    print(_name)
                    

            layer_TS += 1
            layer_DFA += 1
        
        # get the state that satisfied the accepting labels in the DFA
        labels_that_satified_the_task = image_DFA.restrict(self.target_DFA)
        
        # the actual observation will be at the intersection of last set of observation and possible obs
        satis_lbl = (labels_that_satified_the_task & obs_bdd).existAbstract(ts_xcube)

        self.accepting_ts_state = self.obs_bdd.restrict(satis_lbl)

        if verbose:
            ts_S2O_cube_String = self._convert_cube_to_func_S2Obs(bdd_func=satis_lbl)
            print("State Observation before satisfaction")
            for _s in ts_S2O_cube_String:
                _name = self.ts_sym_to_S2obs_map.get(_s)
                assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                print(_name)

            # now extract the set of states that are being expanded during each iteration
            ts_cube_string = self._convert_cube_to_func(bdd_func=self.accepting_ts_state, curr_state_list=self.ts_x_list)
            print("Accepting Abstraction State(s) Reached")
            for _s in ts_cube_string:
                _name = self.ts_sym_to_curr_state_map.get(_s)
                assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                print(_name)

        reached_list_composed[layer_TS] = reached_list_composed[layer_TS] & self.target_DFA & self.accepting_ts_state

        return self.retrive_bfs_wLTL_actions(reached_list_composed, verbose=False)


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
    

    def retrive_bfs_wLTL_actions(self, reached_list_composed, verbose: bool = False):
        """
        Retrieve the plan from symbolic BFS algorithm. The list is sequence of composed states of TS and DFA.
        """
        ts_ycube = reduce(lambda a, b: a & b, self.ts_y_list)
        n = len(reached_list_composed)
        plan = [None for _ in range(n)]
        current = reached_list_composed[n - 1]
        dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)

        for layer in reversed(range(n)):
            # for each action
            # pred = self.manager.bddZero()
            preds: list = []
            for idx, tran_func_action in enumerate(self.ts_transition_fun_list):
                preds.append(self.pre_per_action(trans_action=tran_func_action, From=current, ycube=ts_ycube, x_list=self.ts_x_list, y_list=self.ts_y_list).existAbstract(dfa_xcube))
                # pred = pred | self.pre_per_action(trans_action=tran_func_action, From=current, ycube=ts_ycube, x_list=self.ts_x_list, y_list=self.ts_y_list)

            # pred = pred.existAbstract(dfa_xcube)
            tmp = reached_list_composed[layer - 1].existAbstract(dfa_xcube)

            if verbose:
                # now extract the set of states that are being expanded during each iteration
                ts_cube_string = self._convert_cube_to_func(bdd_func=pred, curr_state_list=self.ts_x_list)
                print("Predeccsors Abstraction State(s)")
                for _s in ts_cube_string:
                    _name = self.ts_sym_to_curr_state_map.get(_s)
                    assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                    print(_name)
                
                ts_cube_string = self._convert_cube_to_func(bdd_func=tmp, curr_state_list=self.ts_x_list)
                print("Predeccsors Abstraction State(s)")
                for _s in ts_cube_string:
                    _name = self.ts_sym_to_curr_state_map.get(_s)
                    assert _name is not None, "Couldn't convert Cube string to its corresponding State. FIX THIS!!!"
                    print(_name)


            for tr_num, pred in enumerate(preds):
                if pred & reached_list_composed[layer - 1].existAbstract(dfa_xcube) != self.manager.bddZero():
                    current = pred & reached_list_composed[layer - 1].existAbstract(dfa_xcube)
                    plan[layer] = tr_num

        # print(plan)

        return plan