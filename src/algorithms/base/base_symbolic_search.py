import sys

from functools import reduce
from typing import Union, List
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


    def convert_add_cube_to_func(self, dd_func: ADD, curr_state_list: List[ADD]) -> List[BDD]:
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
