import sys

from functools import reduce
from typing import Union, List, Optional
from cudd import Cudd, BDD, ADD
from itertools import product


class BaseSymbolicSearch(object):

    def __init__(self,
                 ts_obs_vars: list,
                 cudd_manager: Cudd):
        self.ts_obs_list = ts_obs_vars
        self.manager = cudd_manager
        
    

    def pre(self, From, ycube, x_list: list, y_list: list, transition_fun) -> BDD:
        """
        Compute the predecessors of 'From'.
        
        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        fromY = From.swapVariables(x_list, y_list)
        return transition_fun.andAbstract(fromY, ycube)
    
    def pre_per_action(self, trans_action, From, ycube, x_list: list, y_list: list) -> Union[BDD, ADD]:
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
    

    def image(self, From, xcube, x_list: list, y_list: list, transition_fun) -> Union[BDD, ADD]:
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
    
    
    def image_per_action(self, trans_action, From, xcube, x_list: list, y_list: list) -> Union[BDD, ADD]:
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
        # prod_cubes = self.convert_prod_cube_to_func(key_prod)
        prod_cubes = self.get_prod_cubes(key_prod)

        for key in prod_cubes:
            if key in dict_obj:
                # if key_ts in dict_obj[key_dfa]:
                if not isinstance(dict_obj[key], list):
                    dict_obj[key] = [dict_obj[key]]
                dict_obj[key].append(action)
            else:
                dict_obj[key] = action
    

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


    def convert_add_cube_to_func(self, dd_func: ADD, curr_state_list: List[ADD]) -> List[ADD]:
        """
        A helper function that convert a string of a path to its corresponding boolean form for a Given transition system
        """
        tmp_dd_func = dd_func
        tmp_state_list = curr_state_list

        # Generate cubes for ADD functions generates the vube along with Path value which we ignore for now
        addVars = []
        for cube, _ in tmp_dd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if var == 2 and self.manager.addVar(_idx) not in tmp_state_list:   # not x list is better than y _list because we also have dfa vairables 
                    continue   # skipping over prime states 
                
                elif self.manager.addVar(_idx) in tmp_state_list:
                    if var == 2:
                        _amb_var.append([self.manager.addVar(_idx), ~self.manager.addVar(_idx)])   # count how many vars are missing to fully define the bdd
                    elif var == 0:
                        var_list.append(~self.manager.addVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.addVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                        sys.exit(-1)

            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    addVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                addVars.append(reduce(lambda a, b: a & b, var_list))
        
        # make sure that every element in bddVars container all the bdd variables needed to define it completely
        # for ele in bddVars:
        #     # conver ele to stirng and the bddVars to str and check if all of them exisit or not!
        #     _str_ele = ele.__repr__()
        #     for bVar in tmp_state_list:
        #         assert str(bVar) in _str_ele, "Error! The cube does not contain all the boolean variables in it! FIX THIS!!"

        return addVars

    
    def convert_cube_to_func(self, dd_func: Union[BDD, ADD], curr_state_list: Union[List[ADD], List[BDD]]) -> List[BDD]:
        """
        A helper function that convert a string of a path to its corresponding boolean form for a Given transition system
        """
        if isinstance(dd_func, ADD):
            tmp_dd_func: BDD = dd_func.bddPattern()
            tmp_state_list: List[BDD] = [_avar.bddPattern() for _avar in curr_state_list]
        
        tmp_dd_func = dd_func
        tmp_state_list = curr_state_list
       
        bddVars = []
        for cube in tmp_dd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if var == 2 and self.manager.bddVar(_idx) not in tmp_state_list:   # not x list is better than y _list because we also have dfa vairables 
                    continue   # skipping over prime states 
                
                elif self.manager.bddVar(_idx) in tmp_state_list:
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
            for bVar in tmp_state_list:
                assert str(bVar) in _str_ele, "Error! The cube does not contain all the boolean variables in it! FIX THIS!!"

        return bddVars
    

    def convert_cube_to_func_S2Obs(self, bdd_func: str) -> List[BDD]:
        """
        A helper function that convert a string of a path to its corresponding boolean form for a Given transition system's state observatopn
        """
        if isinstance(bdd_func, ADD):
            bdd_func = bdd_func.bddPattern()

        bddVars = []
        for cube in bdd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if self.manager.bddVar(_idx) in self.ts_obs_list:
                    if var == 2:
                        _amb_var.append([self.manager.bddVar(_idx), ~self.manager.bddVar(_idx)])   # count how many vars are missing to fully define the bdd
                    if var == 0 :
                        var_list.append(~self.manager.bddVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.bddVar(_idx))

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
            for bVar in self.ts_obs_list:
                assert str(bVar) in _str_ele, "Error! The cube does not contain all the boolean variables in it! FIX THIS!!"
        return bddVars

    
    def _look_up_dfa_name(self, prod_dd: BDD, dfa_dict: Union[dict, List[dict]], ADD_flag: bool = False, **kwargs) -> Union[str, List[str]]:
        """
        A helper function that searched through a dictionary (single formula case) or a list of dictionaries (multiple formula case)
        and return the string
        """
        # this will succedd if we have only one dictionary
        try:
            _dfa_dd = prod_dd.existAbstract(self.ts_xcube & self.ts_obs_cube)
            if ADD_flag:
                _dfa_dd = _dfa_dd.bddPattern()
            _dfa_name = dfa_dict.get(_dfa_dd)
            assert _dfa_name is not None, "Couldn't convert DFA Cube to its corresponding State. FIX THIS!!!"
            return _dfa_name
        
        # enter this loop if you have multiple look up dictionaries
        except:
            _dfa_name_list = []
            for idx, _dfa_dict in enumerate(dfa_dict):
                # create a cube of the rest of the dfa vars
                exit_dfa_cube = self.manager.bddOne()
                for cube_idx, cube in enumerate(kwargs['dfa_xcube_list']):
                    if cube_idx != idx:
                        exist_dfa_cube = exit_dfa_cube & cube

                _dfa_dd = prod_dd.existAbstract(self.ts_xcube & self.ts_obs_cube & exist_dfa_cube)
                if ADD_flag:  
                    _dfa_dd = _dfa_dd.bddPattern()              
                _dfa_name = _dfa_dict.get(_dfa_dd)
                assert _dfa_name is not None, "Couldn't convert DFA Cube to its corresponding State. FIX THIS!!!"
                _dfa_name_list.append(_dfa_name)

            return _dfa_name_list
    

    def get_prod_cubes(self, dd_func: Union[BDD, ADD]):
        """
        A function that wraps around convert_prod_cube_to_func and returns a cubes corresponding the product states.
        """
        if isinstance(dd_func, ADD):
            tmp_dd_func: BDD = dd_func.bddPattern()
            tmp_ts_x_list: List[BDD] = [_avar.bddPattern() for _avar in self.ts_x_list]
            tmp_dfa_x_list: List[BDD] = [_avar.bddPattern() for _avar in self.dfa_x_list]
            prod_cube_string: List[Union[BDD, ADD]] = self.convert_prod_cube_to_func(dd_func=tmp_dd_func,
                                                                                     prod_curr_list=tmp_ts_x_list + tmp_dfa_x_list)
        else:
            prod_cube_string: List[Union[BDD, ADD]] = self.convert_prod_cube_to_func(dd_func=dd_func)

        return prod_cube_string


    def get_prod_states_from_dd(self, dd_func: Union[BDD, ADD], obs_flag: bool = False, **kwargs) -> None:
        """
        A function thats wraps arounf convert_cube_to_func() and spits out the states in the corresponding. 

        Set obs_flag to True if you want to print a state's corresponding label/predicate as well. 
        """
        ADD_flag: bool = False

        if isinstance(dd_func, ADD):
            ADD_flag = True
            tmp_dd_func: BDD = dd_func.bddPattern()
            tmp_ts_x_list: List[BDD] = [_avar.bddPattern() for _avar in self.ts_x_list]
            tmp_dfa_x_list: List[BDD] = [_avar.bddPattern() for _avar in self.dfa_x_list]
            prod_cube_string: List[Union[BDD, ADD]] = self.convert_prod_cube_to_func(dd_func=tmp_dd_func,
                                                                                     prod_curr_list=tmp_ts_x_list + tmp_dfa_x_list)
        else:
            prod_cube_string: List[Union[BDD, ADD]] = self.convert_prod_cube_to_func(dd_func=dd_func)

        # prod_cube will have S, Z, P vairables in it. We have to parse it, look the split cubes in their corresponding dictionary individually.
        # print("Prod State(s) Reached")
        for prod_cube in prod_cube_string:
            if ADD_flag:
                prod_cube = prod_cube.toADD()
            # first we extract TS state
            _ts_dd = prod_cube.existAbstract(self.dfa_xcube & self.ts_obs_cube)
            if ADD_flag:
                _ts_dd = _ts_dd.bddPattern()
            # # if ADD_flag:
            #     _ts_name = self.ts_add_sym_to_curr_state_map.get(_ts_dd)
            # else:
            _ts_name = self.ts_bdd_sym_to_curr_state_map.get(_ts_dd)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"
            # Second, we extract DFA state
            # if ADD_flag:
            #     _dfa_name = self._look_up_dfa_name(prod_dd=prod_cube,
            #                                        dfa_dict=self.dfa_add_sym_to_curr_state_map,
            #                                        **kwargs)
            # else:
            _dfa_name = self._look_up_dfa_name(prod_dd=prod_cube,
                                               dfa_dict=self.dfa_bdd_sym_to_curr_state_map,
                                               ADD_flag=ADD_flag,
                                               **kwargs)
            
            if obs_flag:
                # Finally, we extract State label
                _pred_dd = prod_cube.existAbstract(self.prod_xcube)
                _pred_dd = _pred_dd.bddPattern()
                # if ADD_flag:
                #     _pred_name = self.ts_add_sym_to_S2obs_map.get(_pred_dd)
                # else:
                _pred_name = self.ts_bdd_sym_to_S2obs_map.get(_pred_dd)
                assert _pred_name is not None, "Couldn't convert Predicate Cube to its corresponding State. FIX THIS!!!"
                print(f"({_ts_name}, {_pred_name}, {_dfa_name})")
            else:
                print(f"({_ts_name}, {_dfa_name})")

    

    def convert_prod_cube_to_func(self, dd_func: Union[BDD, ADD], prod_curr_list = None) -> List[Union[BDD, ADD]]:
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
        if prod_curr_list is None:
            prod_curr_list = self.ts_x_list + self.dfa_x_list

        ddVars = []
        for cube in dd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                # skip the primed variables
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
                    ddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                ddVars.append(reduce(lambda a, b: a & b, var_list))
        
        return ddVars
