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
    

    def _append_dict_value_simple(self, dict_obj, key_prod, action):
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
    

    def _check_target_dfa_in_closed_list(self, reached_list) -> bool:
        """
        A helper function to check if the DFA target state has been reached or not
        """
        for _, sub_reached_list in reached_list.items():
            if len(sub_reached_list['reached_list']) > 0 and self.target_DFA <= sub_reached_list['reached_list'][-1]:
                return True
        return False
    

    def add_init_state_to_reached_list(self, reached_list: dict) -> str:
        """
        A function that checks if the initial TS state enables any transition on the DFA. If yes, then update that specific DFA state list
        Else, add the initial TS state to the initial DFA state (usually T0_Init). 
        """
        # get the observation of the initial state
        obs_bdd = self.obs_bdd.restrict(self.init_TS)

        # check if any of the DFA edges are satisfied
        image_DFA = self.dfa_transition_fun.restrict(self.init_DFA & obs_bdd)
        image_DFA = image_DFA.swapVariables(self.dfa_y_list, self.dfa_x_list)
        _explicit_dfa_state: str = self.dfa_sym_to_curr_state_map[image_DFA] 
        _init_state = self.dfa_sym_to_curr_state_map[self.init_DFA] 

        # assert _explicit_dfa_state == _init_state, "The initial TS state enable a transition on the DFA. This is not supported by my algorithm yet!"

        reached_list[_explicit_dfa_state]['reached_list'][0] = self.init_TS

        if _explicit_dfa_state == _init_state:
            return self.manager.bddZero()
        else:
            return image_DFA
    

    def composed_symbolic_bfs_wLTL(self, verbose: bool = False):
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
                    self.get_states_from_dd(dd_func=prod_image_bdd, obs_flag=False)
                    
                open_list.append(prod_image_bdd_restricted)
            
            else:
                print("No plan exists! Terminating algorithm.")
                sys.exit(-1)
            
            layer_num += 1
        
        open_list[layer_num] = open_list[layer_num] & self.target_DFA

        if verbose:
            print("********************The goal state encountered is***********************")
            self.get_states_from_dd(dd_func=open_list[layer_num], obs_flag=False)

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

        # while not self.init_TS <= current_prod.existAbstract(self.dfa_xcube):
        for g_layer in reversed(range(max_layer + 1)):
            new_current_prod = self.manager.bddZero()
            for tr_num, prod_tr_action in enumerate(self.composed_tr_list):
                pred_prod= self.pre_per_action(trans_action=prod_tr_action,
                                               From=current_prod,
                                               ycube=self.prod_ycube,
                                               x_list=self.prod_xlist,
                                               y_list=self.prod_ylist)

                if pred_prod & freach_list[g_layer - 1] != self.manager.bddZero():
                    # store the predecessor per action
                    tmp_current_prod = pred_prod & freach_list[g_layer - 1]
                    if verbose:
                        self.get_states_from_dd(dd_func=tmp_current_prod, obs_flag=False)
                    tmp_current_prod_res = (pred_prod & freach_list[g_layer - 1]).existAbstract(self.ts_obs_cube)
                    
                    self._append_dict_value_simple(parent_plan,
                                                   key_prod=tmp_current_prod_res,
                                                   action=self.tr_action_idx_map.inv[tr_num])
                    
                    new_current_prod |= tmp_current_prod_res 
            
            current_prod = new_current_prod

            # assert not current_prod.isZero(), "Error computing the predecessor, Fix this!"

            # g_layer -= 1
        
        return parent_plan




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

        # add the init state to ite respective DFA state. Note, we could start in some other state than the usual T0_init
        init_dfa =  self.add_init_state_to_reached_list(parent_reached_list)

        
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
                    if not closed.isZero() and (parent_reached_list[_dfa_curr_state]['reached_list'][_local_layer_counter] & ~closed ).isZero():
                        # print(f"**************Reached a Fixed Point for DFA State {_dfa_curr_state}**************")
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
                                # check if the state being added has already been explored or not for this DFA state
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
                                                 init_dfa=init_dfa,
                                                 verbose=verbose)
        else:
            print("No plan found")
            sys.exit()
    

    def retrive_bfs_wLTL_actions(self, reached_list_composed, max_layer_num: int, init_dfa: str = '', verbose: bool = False):
        """
        Retrieve the plan from symbolic BFS algorithm. The list is sequence of composed states of TS and DFA.

        Note: Currently our approach can not handle formula where a state multilple predeccsors in the DFA and one of the intermediate edges are triggered.
        I will fix this in future commits. 
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

        if init_dfa.isZero():
            init_dfa = self.init_DFA

        while not(self.init_TS <= current_ts and sym_current_dfa == init_dfa):
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
                    _layer_num = max_layer_num - reverse_count
                    assert _layer_num >= 0, "Trying to access a nonexistent layer. FIX THIS!! "
                    while reached_list_composed[_dfa_state]['reached_list'][_layer_num].isZero():
                        reverse_count += 1
                        _layer_num = max_layer_num - reverse_count
                        assert _layer_num >= 0, "Trying to access a nonexistent layer. FIX THIS!! "

                    valid_pred_ts = self.manager.bddZero() 
                    for tr_num, pred_ts_bdd in enumerate(preds_ts_list):
                        _valid_ts = pred_ts_bdd & reached_list_composed[_dfa_state]['reached_list'][_layer_num] 
                        if verbose:
                            # print the set of states that lie at the intersection of Backward Search and Forward Search
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
                    assert not valid_pred_ts.isZero(), "Error retireving a plan. The intersection of Forward and Backwards Reachable sets should NEVER be empty. FIX THIS!!!"
                    current_ts = valid_pred_ts
                    sym_current_dfa = self.dfa_sym_to_curr_state_map.inv[_dfa_state]
            # normal counter for iteration count
            iter_count += 1
            reverse_count += 1

        return parent_plan