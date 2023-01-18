import re 
import sys
import copy
import random

from functools import reduce
from collections import defaultdict
from typing import List, DefaultDict, Union, Tuple
from bidict import bidict

from cudd import Cudd, BDD, ADD

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import ADDPartitionedDFA
from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs

from utls import *

class AdversarialGame(BaseSymbolicSearch):
    """
     A class that implements optimal strategy synthesis for the Robot player assuming Human player to be adversarial with quantitative constraints. 
    """

    def __init__(self,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 ts_curr_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 sys_act_vars: List[ADD],
                 env_act_vars: List[ADD],
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)

        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.ts_x_list = ts_curr_vars
        self.dfa_x_list = dfa_curr_vars

        self.sys_act_vars = sys_act_vars
        self.env_act_vars = env_act_vars

        self.ts_transition_fun_list: List[List[ADD]] = ts_handle.sym_tr_actions
        self.dfa_transition_fun_list: List[ADD] = dfa_handle.tr_state_adds

        # need these two during preimage computation 
        self.dfa_bdd_x_list = [i.bddPattern() for i in dfa_curr_vars]
        self.dfa_bdd_transition_fun_list: List[BDD] = [i.bddPattern() for i in self.dfa_transition_fun_list]

        self.ts_action_idx_map: bidict = ts_handle.tr_action_idx_map

        self.ts_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        self.ts_sym_to_S2obs_map: bidict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_add_sym_map_curr.inv

        self.ts_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_sym_to_robot_act_map: bidict = ts_handle.predicate_sym_map_robot.inv

        self.obs_add: ADD = ts_handle.sym_state_labels

        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle

        self.winning_states: DefaultDict[int, ADD] = defaultdict(lambda: self.manager.addZero())

        # create corresponding cubes to avoid repetition
        self.ts_xcube: ADD = reduce(lambda x, y: x & y, self.ts_x_list)
        self.dfa_xcube: ADD = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.ts_obs_cube: ADD = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])

        self.sys_cube: ADD = reduce(lambda x, y: x & y, self.sys_act_vars)
        self.env_cube: ADD = reduce(lambda x, y: x & y, self.env_act_vars)

        # create ts and dfa combines cube
        self.prod_xlist: list = self.ts_x_list + self.dfa_x_list
        self.prod_xcube: ADD = reduce(lambda x, y: x & y, self.prod_xlist)

        # sys and env cube
        self.sys_env_cube = reduce(lambda x, y: x & y, [*self.sys_act_vars, *self.env_act_vars])

        self._initialize_w_t()
    
    def _initialize_w_t(self) -> None:
        """
         W : Set of winning states 
        
        This function initializes the set of winning states and set of winning states to the set of accepting states
        """
        ts_states: ADD = self.obs_add
        accp_states: ADD = ts_states & self.target_DFA

        self.winning_states[0] |= accp_states
    

    def _create_lbl_cubes(self):
        """
        A helper function that create cubses of each lbl and store them in a list in the same order as the original order.
         These cubes are used when we convert a BDD to lbl state where we need to extract each lbl.
        """
        sym_lbl_xcube_list = [] 
        for vars_list in self.ts_obs_list:
            sym_lbl_xcube_list.append(reduce(lambda x, y: x & y, vars_list))
        
        return sym_lbl_xcube_list
    

    def _get_max_tr_action_cost(self) -> int:
        """
        A helper function that retireves the highest cost amongst all the transiton function costs
        """
        return max(self.ts_handle.weight_dict.items())
    

    # overriding base class
    def get_prod_states_from_dd(self, dd_func: ADD, **kwargs) -> None:
        """
         This method overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """
        tmp_dd_func: BDD = dd_func.bddPattern()
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=tmp_dd_func, prod_curr_list=kwargs['prod_curr_list']) 
        for prod_cube in prod_cube_string:
            # convert the cube back to 0-1 ADD
            tmp_prod_cube: ADD = prod_cube.toADD()
            _ts_dd = tmp_prod_cube.existAbstract(self.dfa_xcube)
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(_ts_dd, kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=tmp_prod_cube,
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"([{_ts_name}], {_dfa_name})")
    

    def get_prod_state_act_from_dd(self, dd_func: ADD, **kwargs) -> None:
        """
         This method overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """
        tmp_dd_func: BDD = dd_func.bddPattern()
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=tmp_dd_func, prod_curr_list=kwargs['prod_curr_list']) 
        for prod_cube in prod_cube_string:
            # convert the cube back to 0-1 ADD
            tmp_prod_cube: ADD = prod_cube.toADD()
            # get the state ADD
            _ts_dd: ADD = tmp_prod_cube.existAbstract(self.dfa_xcube & self.sys_env_cube)
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(_ts_dd, kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=tmp_prod_cube,
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            # get the Robot action ADD
            ract: ADD = prod_cube.existAbstract(self.prod_xcube & self.ts_obs_cube & self.env_cube)
            ract_name = self.ts_sym_to_robot_act_map[ract]

            # rct_cubes = self.convert_prod_cube_to_func(dd_func=ract, prod_curr_list=self.sys_act_vars)

            hact: ADD = prod_cube.existAbstract(self.prod_xcube & self.ts_obs_cube & self.sys_cube)
            hact_cubes = self.convert_prod_cube_to_func(dd_func=hact, prod_curr_list=self.env_act_vars)
            for dd in hact_cubes:
                hact_name =  self.ts_sym_to_human_act_map[dd]
            
                print(f"({_ts_name}, {_dfa_name})  ----{ract_name} & {hact_name}")
    

    def get_pre_states(self, ts_action: List[BDD], From: BDD, prod_curr_list=None) -> BDD:
        """
         Compute the predecessors using the compositional approach. From is a collection of 0-1 ADD.
          As vectorCompose functionality only works for bdd, we have to first comvert From to 0-1 BDD, 
        """
        pre_prod_state: BDD = self.manager.bddZero()

        # first evolve over DFA and then evolve over the TS
        mod_win_state: BDD = From.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
        
        # for ts_transition in self.ts_transition_fun_list:
        pre_prod_state: BDD = mod_win_state.vectorCompose(prod_curr_list, ts_action)
            
        return pre_prod_state
    

    def solve(self, verbose: bool = False):
        """
         First loop over all the edges in the graph and compute all the possible state values. As the states are reachable from the initial state, 
          all the state should have finite (not infinity) non-zero values except for the accepting state. 
        """
        open_list = defaultdict(lambda: self.manager.addZero())
        # keep tracks all the states action pairs we have visited till now
        closed = self.manager.addZero()
        # computes the max of state action pair max(s, as, -)
        closed_max_min_add: ADD = self.manager.plusInfinity()

        c_max = self._get_max_tr_action_cost()

        # counter used for breaking
        empty_bucket_counter: int = 0

        # counter 
        g_val = self.manager.addZero()
        if g_val.isZero():
            g_layer = 0

        # add the product accepting state to the open list
        open_list[g_layer] = self.winning_states[0]

        prod_curr_list = []
        prod_curr_list.extend([lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])
        prod_curr_list.extend(self.ts_x_list)

        prod_curr_act_list = prod_curr_list + self.sys_act_vars + self.env_act_vars

        prod_bdd_curr_list = [_avar.bddPattern() for _avar in prod_curr_list]
        prod_curr_bdd_act_list = [_avar.bddPattern() for _avar in prod_curr_act_list]

        sym_lbl_cubes = self._create_lbl_cubes()
        
        while not open_list[g_layer].isZero():
            # remove all states that have been explored
            open_list[g_layer] = open_list[g_layer] & ~closed

            # If unexpanded states exist ... 
            if not open_list[g_layer].isZero():
                if verbose:
                    print(f"********************Layer: {g_layer + 1}**************************")
                
                # reset the empty bucket counter 
                empty_bucket_counter = 0
                # Add states to be expanded next to already expanded states
                closed |= open_list[g_layer]

                curr_bucket: ADD = open_list[g_layer].existAbstract(self.sys_env_cube)
                curr_bucket_bdd: BDD = curr_bucket.bddPattern()
                if verbose:
                    print("Computing pre of:")
                    self.get_prod_states_from_dd(dd_func=curr_bucket, sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_bdd_curr_list)

                
                for tr_idx, tr_action in enumerate(self.ts_transition_fun_list):
                    # first get the corresponding transition action cost (constant at the terminal node)
                    curr_act_name: str = self.ts_action_idx_map.inv[tr_idx]
                    if verbose:
                        print(f"Pre under Action: {curr_act_name}")
                    
                    action_cost: ADD =  self.ts_handle.weight_dict[curr_act_name]
                    act_val: int =  int(re.findall(r'\d+', action_cost.__repr__())[0])
                    step_val = g_layer + act_val

                    # compute preimages
                    tr_action_bdd: List[BDD] = [ele.bddPattern() for ele in tr_action]
                    pred_prod = self.get_pre_states(ts_action=tr_action_bdd, From=curr_bucket_bdd, prod_curr_list=prod_bdd_curr_list)

                    if pred_prod.isZero():
                        continue

                    # convert it back to 0-1 ADD 
                    prod_image_restricted = pred_prod.toADD()
                
                    # if verbose:
                        # self.get_prod_states_from_dd(dd_func=prod_image_restricted.existAbstract(self.sys_env_cube), sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_bdd_curr_list)
                    
                    # TODO: Add which action caused this transition to to the BDD before adding it the bucket
                    # if the bucket exists then take the union else initialize the bucket
                    if step_val in open_list:
                        open_list[step_val] |= prod_image_restricted 
                    else:
                        open_list[step_val] = prod_image_restricted
                    

                    if verbose:
                        print(f"*****************Max states-action value under action {curr_act_name} is {step_val}****************************")
                        self.get_prod_states_from_dd(dd_func=prod_image_restricted.existAbstract(self.sys_env_cube), sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_bdd_curr_list)
            
            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == c_max:
                    print("Reached a Fix Point!")
                    break
            
            # g_val = g_val + self.manager.addOne()
            g_layer += 1

            # compute the max of state and action pair
            closed_max_add = closed_max_add.max(open_list[g_layer].ite(self.manager.addConst(g_layer), self.manager.addZero()))
            
            # if verbose:
            #     print(f"*****************Max states-action value at layer {g_layer}****************************")
            #     # self.get_prod_state_act_from_dd(dd_func=open_list[g_layer].existAbstract(self.sys_env_cube),
            #     #                                 sym_lbl_cubes=sym_lbl_cubes,
            #     #                                 prod_curr_list=prod_curr_bdd_act_list)
            #     self.get_prod_states_from_dd(dd_func=open_list[g_layer].existAbstract(self.sys_env_cube), sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_bdd_curr_list)

            if g_layer > max(open_list.keys()):
                g_layer -= 1
                break

            # keep updating g_layer up until the most recent bucket
            while g_layer not in open_list:
                # g_val = g_val + self.manager.addOne()
                g_layer += 1
                
        print(f"********************Took {g_layer} layers to reach a fixed point********************")

        return
