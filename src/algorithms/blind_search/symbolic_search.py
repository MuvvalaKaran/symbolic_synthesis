'''
This file implements Symbolic Graph search algorithms
'''
import sys

from bidict import bidict
from functools import reduce
from cudd import Cudd, BDD, ADD

from typing import List
from itertools import product

from src.symbolic_graphs import SymbolicDFA, SymbolicTransitionSystem


class SymbolicSearch(object):
    """
    Given a Graph, find the shortest path as per the symbolic A* (BDDA*) algorithm as outlined by Jensen, Bryant, Valeso's paper.
    """

    def __init__(self,
                 ts_handle: SymbolicTransitionSystem,
                 dfa_handle: SymbolicDFA,
                 manager: Cudd,
                 ts_curr_vars: list,
                 ts_next_vars: list,
                 ts_obs_vars: list,
                 dfa_curr_vars: list,
                 dfa_next_vars: list):

        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state
        self.manager = manager
        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_obs_list = ts_obs_vars
        self.ts_transition_fun_list = ts_handle.sym_tr_actions
        self.dfa_transition_fun = dfa_handle.dfa_bdd_tr
        self.ts_sym_to_curr_state_map: dict = ts_handle.predicate_sym_map_curr.inv
        self.ts_sym_to_S2obs_map: dict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_sym_map_curr.inv
        self.dfa_sym_to_nxt_state_map: dict = dfa_handle.dfa_predicate_sym_map_nxt.inv
        self.obs_bdd = ts_handle.sym_state_labels
        self.tr_action_idx_map = ts_handle.tr_action_idx_map

        # create corresponding cubes to avoid repetition
        self.ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        self.ts_ycube = reduce(lambda x, y: x & y, self.ts_y_list)
        self.ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)

        self.dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)

        # composed graph consists of state S, Z and hence are function TS and DFA bdd vars
        self.prod_xlist = self.ts_x_list + self.dfa_x_list
        self.prod_ylist = self.ts_y_list + self.dfa_y_list
        self.prod_xcube = reduce(lambda x, y: x & y, self.prod_xlist)
        self.prod_ycube = reduce(lambda x, y: x & y, self.prod_ylist)

        # composed monolithic TR
        self.composed_tr_list = self._construct_composed_tr_function()

    
    def pre(self, From, ycube, x_list: list, y_list: list, transition_fun):
        """
        Compute the predecessors of 'From'.
        
        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        fromY = From.swapVariables(x_list, y_list)
        return transition_fun.andAbstract(fromY, ycube)
    
    def pre_per_action(self, trans_action, From, ycube, x_list: list, y_list: list):
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
    

    def _append_dict_value_composed(self, dict_obj, key_prod, action):
        """
        Check if key exist in dict or not.

        If Key exist in dict:
           Check if type of value of key is list or not
           
           If type is not list then make it list and Append the value in list
        
        else: add key-value pair
        """
        # compute all the cubes and store them individually
        prod_cubes = self.convert_prod_cube_to_func(key_prod)

        for key in prod_cubes:
            if key in dict_obj:
                # if key_ts in dict_obj[key_dfa]:
                if not isinstance(dict_obj[key], list):
                    dict_obj[key] = [dict_obj[key]]
                dict_obj[key].append(action)
            else:
                dict_obj[key] = action
    
    
    def _construct_composed_tr_function(self) -> List[BDD]:
        """
        A function that construct that conjoints the TR function of the TS and DFA along with S2P (state to obs BDD).

        Note: We prime the S2P BDD because we want to extract the next state in the DFA after we evolve over the TS.
        """

        obs_bdd_prime = self.obs_bdd.swapVariables(self.ts_x_list, self.ts_y_list) 
        composed_tr_list = []
        for tr_action in self.ts_transition_fun_list:
            composed_tr = tr_action & obs_bdd_prime & self.dfa_transition_fun
            composed_tr_list.append(composed_tr)
        
        return composed_tr_list
    

    def get_states_from_dd(self, dd_func: BDD, obs_flag: bool = False) -> None:
        """
        A function thats wraps arounf convert_cube_to_func() and spits out the states in the corresponding. 

        Set obs_flag to True if you want to print a state's corresponding label/predicate as well. 
        """
        
        prod_cube_string = self.convert_prod_cube_to_func(bdd_func=dd_func)

        # prod_cube will have S, Z, P vairables in it. We have to parse it, look the split cubes in their corresponding dictionary individually.
        print("Prod State(s) Reached")
        for prod_cube in prod_cube_string:
            # continue
            # first we extract TS state
            _ts_bdd = prod_cube.existAbstract(self.dfa_xcube & self.ts_obs_cube)
            _ts_name = self.ts_sym_to_curr_state_map.get(_ts_bdd)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"
            # Second, we extract DFA state
            _dfa_bdd = prod_cube.existAbstract(self.ts_xcube & self.ts_obs_cube)
            _dfa_name = self.dfa_sym_to_curr_state_map.get(_dfa_bdd)
            assert _dfa_name is not None, "Couldn't convert DFA Cube to its corresponding State. FIX THIS!!!"
            if obs_flag:
                # Finally, we extract State label
                _pred_bdd = prod_cube.existAbstract(self.prod_xcube)
                _pred_name = self.ts_sym_to_S2obs_map.get(_pred_bdd)
                assert _pred_name is not None, "Couldn't convert Predicate Cube to its corresponding State. FIX THIS!!!"

                print(f"({_ts_name}, {_pred_name}, {_dfa_name})")
            else:
                print(f"({_ts_name}, {_dfa_name})")

    

    def convert_prod_cube_to_func(self, bdd_func: BDD) -> List[BDD]:
        """
        A helper function to determine the product state (S, Z) and the corresponding observation (S->P) from a given BDD

        NOTE: A partitioned convert_cube_to_func for each Attribute, i.e., S, Z, P will not work for readibility as we dont know which is the right 
        (S, Z, P) pair. For e.g.
        
         For fomula phi = F(l2) & F(l6) in 5x5 fridworld, Z corresponding to l2 and l6 is T0_S1 and T0_S3 respectibely. When we compute
         the image of (s2, T0_S1) and (s6, T0_S3), we encounter states (s1, T0_S1); (s1, T0_S3); (s7, T0_S1); (s7, T0_S3) amomngst other.
         Thus, we need a didcated function to identity the pairs individually. 
        
        Unlike other convert_to_cube function, this one does not check for a variable in the state list as the bdd function is fully defined
        over the prod graph, i.e., we check variables that belong to S (TS state), Z (DFA state), and Prodicate (P)/ observations
        """
        # we dont add bdd Vars related P as they do have their prime counterparts and hence,
        #  we will always include these vairbales in our cube construnction
        prod_curr_list = self.ts_x_list + self.dfa_x_list

        bddVars = []
        for cube in bdd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                # skip the primaed variables
                if var == 2 and self.manager.bddVar(_idx) not in prod_curr_list:   # not x list is better than y _list because we also have dfa vairables 
                    continue   # skipping over prime states 
                else:
                    if var == 2:
                        _amb_var.append([self.manager.bddVar(_idx), ~self.manager.bddVar(_idx)])   # count how many vars are missing to fully define the bdd
                    elif var == 0:
                        var_list.append(~self.manager.bddVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.bddVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                        sys.exit(-1)
                
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    bddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                bddVars.append(reduce(lambda a, b: a & b, var_list))
        return bddVars
    

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
    

    def composed_symbolic_bfs_wLTL(self, verbose: bool = False, obs_flag: bool = False):
        """
        A function that compose the TR function from the Transition system and DFA and search symbolically over the product graph.

        Note: Since we are evolving concurrently over the prod DFA, the state labeling update is on step behind, i.e.,
         when we compute image over the 
        """
        # while open list is a sequence of bucket closed is a just a set of explored states and hence is not numbered
        open_list = []
        closed = self.manager.bddZero()

        # maintain a common layering number
        layer_num = 0
                
        # add the init state to ite respective DFA state. Note, we could start in some other state than the usual T0_init
        open_list.append(self.init_TS & self.init_DFA)

        while not self.target_DFA <= open_list[layer_num].existAbstract(self.ts_xcube):
            # remove all states that have been explored
            open_list[layer_num] = open_list[layer_num] & ~closed

            # If unexpanded states exist ... 
            if not open_list[layer_num].isZero():
                if verbose:
                    print(f"********************Layer: {layer_num}**************************")
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer_num]

                # compute the image of the TS states 
                prod_image_bdd = self.manager.bddZero()
                for prod_tr_action in self.composed_tr_list:
                    image_prod = self.image_per_action(trans_action=prod_tr_action,
                                                       From=open_list[layer_num],
                                                       xcube=self.prod_xcube,
                                                       x_list=self.prod_xlist,
                                                       y_list=self.prod_ylist)
                    
                    prod_image_bdd |= image_prod

                prod_image_bdd_restricted = prod_image_bdd.existAbstract(self.ts_obs_cube)
                
                if verbose:
                    self.get_states_from_dd(dd_func=prod_image_bdd, obs_flag=obs_flag)
                    
                open_list.append(prod_image_bdd_restricted)
            
            else:
                print("No plan exists! Terminating algorithm.")
                sys.exit(-1)
            
            layer_num += 1
        
        open_list[layer_num] = open_list[layer_num] & self.target_DFA

        if verbose:
            print("********************The goal state encountered is***********************")
            self.get_states_from_dd(dd_func=open_list[layer_num], obs_flag=obs_flag)

        print(f"Found a plan with least cost length {layer_num}, Now retireving it!")

        return self.retrieve_composed_symbolic_bfs(max_layer=layer_num, freach_list=open_list, verbose=verbose)
    

    def retrieve_composed_symbolic_bfs(self, max_layer: int, freach_list: dict, verbose: bool = False):
        """
        Retrieve the plan through Backward search by strarting from the Goal state and computing the interseaction of Forwards and Backwards
         Reachable set. 
        """
        g_layer = max_layer
        print("Working Retrieval plan now")

        current_prod = freach_list[g_layer]

        parent_plan = {}

        for g_layer in reversed(range(max_layer + 1)):
            new_current_prod = self.manager.bddZero()
            for tr_num, prod_tr_action in enumerate(self.composed_tr_list):
                pred_prod= self.pre_per_action(trans_action=prod_tr_action,
                                               From=current_prod,
                                               ycube=self.prod_ycube,
                                               x_list=self.prod_xlist,
                                               y_list=self.prod_ylist)

                if pred_prod.isZero():
                    continue
                
                if pred_prod & freach_list[g_layer - 1] != self.manager.bddZero():
                    # store the predecessor per action
                    tmp_current_prod = pred_prod & freach_list[g_layer - 1]
                    if verbose:
                        self.get_states_from_dd(dd_func=tmp_current_prod, obs_flag=False)
                    tmp_current_prod_res = (pred_prod & freach_list[g_layer - 1]).existAbstract(self.ts_obs_cube)
                    
                    self._append_dict_value_composed(parent_plan,
                                                     key_prod=tmp_current_prod_res,
                                                     action=self.tr_action_idx_map.inv[tr_num])
                    
                    new_current_prod |= tmp_current_prod_res 
            
            current_prod = new_current_prod
        
        return parent_plan


class SymbolicSearchFranka(SymbolicSearch):
    """
     This clas overrides the base class's printing funtionality to print predicate names
    """
    def __init__(self,
                 ts_handle: SymbolicTransitionSystem,
                 dfa_handle: SymbolicDFA,
                 manager: Cudd,
                 ts_curr_vars: list,
                 ts_next_vars: list, 
                 ts_obs_vars: list, 
                 dfa_curr_vars: list, 
                 dfa_next_vars: list):

        super().__init__(ts_handle, dfa_handle, manager, ts_curr_vars, ts_next_vars, ts_obs_vars, dfa_curr_vars, dfa_next_vars)
        self.pred_int_map: dict = ts_handle.pred_int_map
    

    def get_state_from_tuple(self, state_tuple: tuple) -> List[str]:
        """
         Given, a predicate tuple, this function return the corresponding state tuple
        """
        if isinstance(state_tuple, tuple):
            _states = [self.pred_int_map.inv[state] for state in state_tuple]
        else:
            _states = self.pred_int_map.inv[state_tuple]

        return _states


    def get_states_from_dd(self, dd_func: BDD) -> None:
        """
        A function thats wraps arounf convert_cube_to_func() and spits out the states in the corresponding. 

        Set obs_flag to True if you want to print a state's corresponding label/predicate as well. 
        """
        
        prod_cube_string = self.convert_prod_cube_to_func(bdd_func=dd_func)

        # prod_cube will have S, Z, P vairables in it. We have to parse it, look the split cubes in their corresponding dictionary individually.
        print("Prod State(s) Reached")
        for prod_cube in prod_cube_string:
            # first we extract TS state
            _ts_bdd: BDD = prod_cube.existAbstract(self.dfa_xcube & self.ts_obs_cube)
            _ts_tuple: tuple = self.ts_sym_to_curr_state_map.get(_ts_bdd)
            assert _ts_tuple is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"
            _ts_name: str =  self.get_state_from_tuple(state_tuple=_ts_tuple)

            # Second, we extract DFA state
            _dfa_bdd = prod_cube.existAbstract(self.ts_xcube & self.ts_obs_cube)
            _dfa_name = self.dfa_sym_to_curr_state_map.get(_dfa_bdd)
            assert _dfa_name is not None, "Couldn't convert DFA Cube to its corresponding State. FIX THIS!!!"
            print(f"({_ts_name}, {_dfa_name})")