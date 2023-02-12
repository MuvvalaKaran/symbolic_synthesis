'''
 This scripts all the functions to construct the graph of utility purely in symbolic Fashion. 
'''
import re
import sys
import time
import warnings

from math import inf
from bidict import bidict
from functools import reduce
from itertools import product
from collections import defaultdict
from typing import Union, List, Tuple, Dict

from cudd import Cudd, BDD, ADD

from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs
from src.symbolic_graphs import ADDPartitionedDFA


class SymbolicGraphOfUtility(DynWeightedPartitionedFrankaAbs):
    
    def __init__(self,
                 curr_vars: list,
                 lbl_vars: list,
                 state_utls_vars: List[ADD],
                 robot_action_vars: list,
                 human_action_vars: list,
                 task,
                 domain,
                 ts_state_map: dict,
                 ts_states: list,
                 manager: Cudd,
                 weight_dict: dict,
                 seg_actions: dict,
                 ts_state_lbls: list,
                 dfa_state_vars: List[ADD],
                 sup_locs: List[str],
                 top_locs: List[str],
                 budget: int,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 int_weight_dict: Dict[str, int],
                 **kwargs):
        super().__init__(curr_vars,
                         lbl_vars,
                         robot_action_vars,
                         human_action_vars,
                         task,
                         domain,
                         ts_state_map,
                         ts_states,
                         manager,
                         weight_dict,
                         seg_actions,
                         ts_state_lbls,
                         dfa_state_vars,
                         sup_locs,
                         top_locs,
                         **kwargs)
        self.ts_handle: DynWeightedPartitionedFrankaAbs = ts_handle

        self.int_weight_dict = int_weight_dict

        self.sym_vars_ults: List[ADD] = state_utls_vars
        self.energy_budget: int = budget

        # index to determine where the state vars start 
        self.state_start_idx: int =  len(self.sym_vars_human) + len(self.sym_vars_robot) + len(dfa_state_vars) + sum([len(listElem) for listElem in self.sym_vars_lbl]) +  len(self.sym_vars_curr)

        self.predicate_sym_map_utls: bidict = {}
        self.utls_cube = reduce(lambda x, y: x & y, self.sym_vars_ults)

        num_of_acts = len(list(self.tr_action_idx_map.keys()))
        self.sym_tr_actions = [[self.manager.addZero() for _ in range(len(state_utls_vars))] for _ in range(num_of_acts)]

        # create state util look up dictionary
        self._initialize_add_for_state_utls()
    

    def _initialize_add_for_state_utls(self):
        """
         This function initializes ADD that represents the Prod state utility value at each state. 
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_ults)))
        _node_int_map = bidict({_hint: boolean_str[_hint] for _hint in range(self.energy_budget + 1)})

        assert len(boolean_str) >= len(_node_int_map), \
             "FIX THIS: Looks like there are more state utility values than it's corresponding boolean variables!"
        
        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map.items():
            _val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _val_list.append(self.sym_vars_ults[_idx])
                else:
                    _val_list.append(~self.sym_vars_ults[_idx])
            
            _bool_func_curr = reduce(lambda a, b: a & b, _val_list)

            # update bidict accordingly
            _node_int_map[_key] = _bool_func_curr
        
        self.predicate_sym_map_utls = bidict(_node_int_map)
    

    def add_utl_edg_to_tr(self,
                          curr_utls_sym: ADD,
                          nxt_utls_sym: ADD,
                          robot_action_name: str,
                          mod_act_dict: dict):
        """
         A function that build the TR that captures the evolution of utility for a given robot act.
        """
        # get the modified robot action name
        mod_raction_name: str = mod_act_dict[robot_action_name]
        robot_move: ADD = self.ts_handle.predicate_sym_map_robot[mod_raction_name]

        _tr_idx: int = self.tr_action_idx_map.get(robot_action_name)

        # generate all the cubes, with their corresponding string repr and leaf value (state value should be 1)
        add_cube: List[Tuple(list, int)] = list(nxt_utls_sym.generate_cubes())   
        assert len(add_cube) == 1, "Error computing cube string for next state's symbolic representation. FIX THIS!!!"
        assert add_cube[0][1] == 1, "Error computing next state cube. The integer value of the leaf node in the ADD is not 1. FIX THIS!!!"

        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(add_cube[0][0]):
            if var == 1 and self.manager.addVar(_idx) in self.sym_vars_ults:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                
                self.sym_tr_actions[_tr_idx][_state_idx] |= curr_utls_sym & robot_move

            
            elif var == 2 and self.manager.addVar(_idx) in self.sym_vars_ults:
                warnings.warn("Encountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)
        
        
        
    

    def create_sym_tr_actions(self, mod_act_dict: dict, verbose: bool = False) -> None:
        """
          A function that loops over all the utility variables and construts the TR (the evolution of the utiltiy value) for every action.  
        """
        # for each utility value. . .
        for curr_utl, curr_utl_dd in self.predicate_sym_map_utls.items():
            # we do evolve after reach the max utility value
            if curr_utl == self.energy_budget:
                continue
            
            # for each action, construct the TR
            for ts_action, ts_cost in self.int_weight_dict.items():
                if 'human' in ts_action:
                    continue

                # successor utl value
                succ_utl: int = curr_utl +  ts_cost

                if succ_utl > self.energy_budget:
                    continue

                succ_utl_dd: ADD = self.predicate_sym_map_utls[succ_utl]

                # add edge curr_utl -----{robot_act}-----> succ_utl, human action is set to True
                self.add_utl_edg_to_tr(curr_utls_sym=curr_utl_dd,
                                       nxt_utls_sym=succ_utl_dd,
                                       robot_action_name=ts_action,
                                       mod_act_dict=mod_act_dict)


                if verbose:
                    print(f"Adding edge {curr_utl} ------{ts_action}-----> {succ_utl}")
