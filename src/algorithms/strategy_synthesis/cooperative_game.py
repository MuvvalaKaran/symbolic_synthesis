import re 
import sys
import math 
import copy
import random

from functools import reduce
from itertools import product
from collections import defaultdict
from typing import List, DefaultDict, Union, Tuple, Dict
from bidict import bidict


from cudd import Cudd, BDD, ADD

from src.algorithms.strategy_synthesis import AdversarialGame
from src.symbolic_graphs import ADDPartitionedDFA
from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs

from utls import *


class CooperativeGame(AdversarialGame):
    """
     A class the computes the optimal strategy assuming the Human to cooperative, i.e. both players are playing Min-Min
    """
    
    def __init__(self,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 ts_curr_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 sys_act_vars: List[ADD],
                 env_act_vars: List[ADD],
                 cudd_manager: Cudd) -> None:
        super().__init__(ts_handle,
                         dfa_handle,
                         ts_curr_vars,
                         dfa_curr_vars,
                         ts_obs_vars,
                         sys_act_vars,
                         env_act_vars,
                         cudd_manager)
    

    def solve(self, verbose: bool = False) -> ADD:
        """
         Method that computes the optimal strategy for the system with minimum payoff under cooperative environment assumptions. 
        """
        ts_states: ADD = self.obs_add
        accp_states: ADD = ts_states & self.target_DFA

        # convert it to 0 - Infinity ADD
        accp_states = accp_states.ite(self.manager.addZero(), self.manager.plusInfinity())
        
        # strategy - optimal (state & robot-action) pair stored in the ADD
        strategy: ADD  = self.manager.plusInfinity()

        # initializes accepting states to be zero
        self.winning_states[0] |= self.winning_states[0].min(accp_states)
        strategy = strategy.min(accp_states)

        layer: int = 0
        c_max: int = self._get_max_tr_action_cost()

        prod_curr_list = []
        prod_curr_list.extend([lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])
        prod_curr_list.extend(self.ts_x_list)
        
        prod_bdd_curr_list = [_avar.bddPattern() for _avar in prod_curr_list]

        sym_lbl_cubes = self._create_lbl_cubes()

        while True: 
            if self.winning_states[layer].compare(self.winning_states[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                init_state_cube = list(((self.init_TS & self.init_DFA) & self.winning_states[layer]).generate_cubes())[0]
                init_val = init_state_cube[1]
                self.init_state_value = init_val
                if init_val != math.inf:
                    print(f"A Winning Strategy Exists!!. The Min Energy is {init_val}")
                    return tmp_strategy
                else:
                    print("No Winning Strategy Exists!!!")
                    # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
                    del self.winning_states
                    return
            
            print(f"**************************Layer: {layer}**************************")

            _win_state_bucket: Dict[BDD] = defaultdict(lambda: self.manager.bddZero())

            
            # convert the winning states into buckets of BDD
            _max_interval_val = layer * c_max
            for sval in range(_max_interval_val + 1):
                # get the states with state value equal to sval and store them in their respective bukcets
                win_sval = self.winning_states[layer].bddInterval(sval, sval)
                
                if not win_sval.isZero():
                    _win_state_bucket[sval] |= win_sval
            
            _pre_buckets: Dict[ADD] = defaultdict(lambda: self.manager.addZero())

            # compute the predecessor and store them by action cost + successor cost
            for tr_idx, tr_action in enumerate(self.ts_bdd_transition_fun_list):
                curr_act_name: str = self.ts_action_idx_map.inv[tr_idx]
                action_cost: ADD =  self.ts_handle.weight_dict[curr_act_name]
                act_val: int = list(action_cost.generate_cubes())[0][1]
                for sval, succ_states in _win_state_bucket.items():
                    pre_states: BDD = self.get_pre_states(ts_action=tr_action, From=succ_states, prod_curr_list=prod_bdd_curr_list)

                    if not pre_states.isZero():
                        _pre_buckets[act_val + sval] |= pre_states.toADD()
            
            # unions of all predecessors
            pre_states: ADD = reduce(lambda x, y: x | y, _pre_buckets.values())

            # print non-zero states in this iteration
            # if verbose:
            #     print(f"Non-Zero states at Iteration {layer + 1}")
            #     self.get_prod_states_from_dd(dd_func=upre_states.existAbstract(self.sys_cube), sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_dfa_bdd_curr_list)
            
            # We need to take the unions of all the (s, a_s, a_e). But, we tmp_strategy to have background vale of inf, useful later when we take max() operation.
            # Thus, I am using this approach. 
            tmp_strategy: ADD = pre_states.ite(self.manager.addOne(), self.manager.plusInfinity())

            # accomodate for worst-case human behavior
            for sval, apre_s in _pre_buckets.items():
                tmp_strategy = tmp_strategy.max(apre_s.ite(self.manager.addConst(int(sval)), self.manager.addZero()))

            new_tmp_strategy: ADD = tmp_strategy

            # go over all the human actions and preserve the minimum one
            for human_tr_dd in self.ts_sym_to_human_act_map.keys():
                new_tmp_strategy = new_tmp_strategy.min(tmp_strategy.restrict(human_tr_dd)) 

            # compute the minimum of state action pairs
            strategy = strategy.min(new_tmp_strategy)

            self.winning_states[layer + 1] |= self.winning_states[layer]

            for tr_dd in self.ts_sym_to_robot_act_map.keys():
                # remove the dependency for that action and preserve the minimum value for every state
                self.winning_states[layer + 1] = self.winning_states[layer  + 1].min(strategy.restrict(tr_dd))

            if verbose:
                print(f"Minimum State value at Iteration {layer +1}")
                self.get_state_value_from_dd(dd_func=self.winning_states[layer + 1], sym_lbl_cubes=sym_lbl_cubes, prod_list=[*prod_curr_list, *self.dfa_x_list])
            
            # update counter 
            layer += 1
    

    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to roll out the synthesized Min-Min strategy
        """
        curr_dfa_state = self.init_DFA
        curr_prod_state = self.init_TS & curr_dfa_state
        counter = 0
        max_layer: input = max(self.winning_states.keys())
        ract_name: str = ''

        sym_lbl_cubes = self._create_lbl_cubes()

        if verbose:
            init_ts = self.init_TS
            init_ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=init_ts, sym_lbl_xcube_list=sym_lbl_cubes)
            init_ts = self.ts_handle.get_state_from_tuple(state_tuple=init_ts_tuple)
            print(f"Init State: {init_ts[1:]}")     

        # until you reach a goal state. . .
        while (self.target_DFA & curr_prod_state).isZero():
            # current state tuple 
            curr_ts_state: ADD = curr_prod_state.existAbstract(self.dfa_xcube & self.sys_env_cube).bddPattern().toADD()   # to get 0-1 ADD
            curr_ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=curr_ts_state, sym_lbl_xcube_list=sym_lbl_cubes)
            
            # get the state with its minimum state value
            opt_sval_cube =  list((curr_prod_state & self.winning_states[max_layer]).generate_cubes())[0]
            curr_prod_state_sval: int = opt_sval_cube[1]

            # this gives us a sval-infinity ADD
            curr_state_act_cubes: ADD =  strategy.restrict(curr_prod_state)
            # get the 0-1 version
            act_cube: ADD = curr_state_act_cubes.bddInterval(curr_prod_state_sval, curr_prod_state_sval).toADD()

            # if there are multuple human edges then do not intervene else follow the unambiguous human action from the strategy
            sys_act_cube = act_cube.bddPattern().existAbstract(self.env_cube.bddPattern()).toADD()
            list_act_cube = self.convert_add_cube_to_func(sys_act_cube, curr_state_list=self.sys_act_vars)

            # if multiple winning actions exisit from same state
            if len(list_act_cube) > 1:
                ract_name = None
                while ract_name is None:
                    sys_act_cube: List[int] = random.choice(list_act_cube)
                    ract_name = self.ts_sym_to_robot_act_map.get(sys_act_cube, None)

            else:
                ract_name = self.ts_sym_to_robot_act_map[sys_act_cube]

            if verbose:
                print(f"Step {counter}: Conf: {self.ts_handle.get_state_from_tuple(curr_ts_tuple)} Act: {ract_name}")
            
            # look up the next tuple 
            next_exp_states = self.ts_handle.adj_map[curr_ts_tuple][ract_name]['r']
            next_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)

            # for a given (s, a_s) there should be exaclty one human action (a_e). 
            # If you encounter multiple then that corresponds to no-human intervention
            curr_state_human_act_cubes: ADD = strategy.restrict(curr_prod_state & sys_act_cube)
            print(f"{counter}: State value: {curr_prod_state_sval}")
            human_act_cube: ADD = curr_state_human_act_cubes.bddInterval(curr_prod_state_sval, curr_prod_state_sval).toADD()
            # print(human_act_cube.display())
            human_list_act_cube = self.convert_add_cube_to_func(human_act_cube, curr_state_list=self.env_act_vars)

            # This logic does not work for Cooperative games - just check the state value during roll out
            if len(human_list_act_cube) > 1:
                for env_act_cube in human_list_act_cube:
                    full_state_act_cube = strategy.restrict(curr_prod_state & sys_act_cube & env_act_cube)
                    
                    if full_state_act_cube.findMin() == self.manager.addConst(opt_sval_cube[1]):
                        hact_name = self.ts_sym_to_human_act_map.get(env_act_cube, ' ')
                        # check is this robot action exisits from (s, a_s) in the adj dictionary
                        if self.ts_handle.adj_map.get(curr_ts_tuple, {}).get(ract_name, {}).get(hact_name):
                            next_exp_states = self.ts_handle.adj_map[curr_ts_tuple][ract_name][hact_name]
                            next_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)
                            
                            if verbose:
                                print(f"Human Moved: New Conf.: {self.ts_handle.get_state_from_tuple(next_tuple)} Act: {hact_name}")
                            
                            break
            else:
                hact_name = self.ts_sym_to_human_act_map[human_act_cube]
                # look up the next tuple - could fail if the no-intervention edge is an unambiguous one.  
                if self.ts_handle.adj_map.get(curr_ts_tuple, {}).get(ract_name, {}).get(hact_name):
                    next_exp_states = self.ts_handle.adj_map[curr_ts_tuple][ract_name][hact_name]
                    next_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)

                    if verbose:
                        print(f"Human Moved: New Conf.: {self.ts_handle.get_state_from_tuple(next_tuple)} Act: {hact_name}")

            # look up its corresponding formula
            curr_ts_state: ADD = self.ts_handle.get_sym_state_from_tuple(next_tuple)

            # convert the 0-1 ADD to BDD for DFA edge checking
            curr_ts_lbl: BDD = curr_ts_state.existAbstract(self.ts_xcube).bddPattern()

            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state in self.dfa_sym_to_curr_state_map.keys():
                bdd_dfa_state: BDD = dfa_state.bddPattern()
                dfa_pre: BDD = bdd_dfa_state.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
                edge_exists: bool = not (dfa_pre & (curr_dfa_state.bddPattern() & curr_ts_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state: BDD = dfa_state
                    break
            
            curr_prod_state: ADD = curr_ts_state & curr_dfa_state

            counter += 1
        
        # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
        del self.winning_states