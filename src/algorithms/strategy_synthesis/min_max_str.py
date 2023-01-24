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
        self.ts_bdd_transition_fun_list: List[List[BDD]] = []

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

        # ADD that keeps track of the optimal values of state at each iteration
        self.winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())

        # get the bdd version of the transition function as vectorComposition only works with
        for act in self.ts_transition_fun_list:
            act_ls = []
            for avar in act:
                act_ls.append(avar.bddPattern())
        
            self.ts_bdd_transition_fun_list.append(act_ls)
    

    def _create_lbl_cubes(self) -> List[ADD]:
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
        _max = 0
        for tr_action in self.ts_handle.weight_dict.values():
            if not tr_action.isZero():
                action_cost = tr_action.findMax()
                action_cost_int = int(re.findall(r'\d+', action_cost.__repr__())[0])
                if action_cost_int > _max:
                    _max = action_cost_int
        
        return _max

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
    

    def get_state_value_from_dd(self, dd_func: ADD, prod_list: List[ADD], **kwargs) -> None:
        """
         A helper function that print the value associated with each state.
          
          @param: dd_func is wining_state ADD with minimum value at any given iteration
        """

        addVars = []
        for cube, sval in dd_func.generate_cubes():
            if sval == math.inf:
                continue
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if var == 2 and self.manager.addVar(_idx) not in prod_list:
                    continue
                
                elif self.manager.addVar(_idx) in prod_list:
                    if var == 2:
                        _amb_var.append([self.manager.addVar(_idx), ~self.manager.addVar(_idx)])
                    elif var == 0:
                        var_list.append(~self.manager.addVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.addVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                        sys.exit(-1)

            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    addVars.append((reduce(lambda a, b: a & b, var_list), sval))
                    var_list = list(set(var_list) - set(_ele))
            else:
                addVars.append((reduce(lambda a, b: a & b, var_list), sval))
        
        
        for cube, val in addVars: 
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(cube.existAbstract(self.dfa_xcube), kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=cube,
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"({_ts_name}, {_dfa_name}) {val}")

    

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
    

    def get_pre_states(self, ts_action: List[BDD], From: BDD, prod_curr_list=None, act_name: str = '') -> BDD:
        """
         Compute the predecessors using the compositional approach. From is a collection of 0-1 ADD.
          As vectorCompose functionality only works for bdd, we have to first comvert From to 0-1 BDD, 
        """
        # first evolve over DFA and then evolve over the TS
        # mod_win_state: BDD = From.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
        
        # pre_prod_state: BDD = mod_win_state.vectorCompose(prod_curr_list, ts_action)

        # testing this new one 
        pre_prod_state_2: BDD = From.vectorCompose([*self.dfa_bdd_x_list, *prod_curr_list], [*self.dfa_bdd_transition_fun_list, *ts_action])

        # if act_name == 'release':
        #     if (pre_prod_state & self.init_DFA.bddPattern()).isZero():
        #         print("Error computing pre of Release action. FIX THIS!!!")
            
        #     if (pre_prod_state_2 & self.init_DFA.bddPattern()).isZero():
        #         print("Error computing pre(2) of Release action. FIX THIS!!!")
        
        # if act_name == 'transit b0':
        #     if (pre_prod_state & self.init_DFA.bddPattern()).isZero():
        #         print("Error computing pre(1) of transit b0 action. FIX THIS!!!")
            
        #     if (pre_prod_state_2 & self.init_DFA.bddPattern()).isZero():
        #         print("Error computing pre(2) of transit b0 action. FIX THIS!!!")

        # sys.exit(-1)
        # assert pre_prod_state_2 == pre_prod_state, f"Error computing Predecessor states under action {act_name}. FIX THIS!!!"
            
        # return pre_prod_state
        return pre_prod_state_2
    

    def solve(self, verbose: bool = False):
        """
         Method that compute the optimal strategy for the system with minimum payoff under adversarial environment assumptions. 
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

        prod_dfa_bdd_curr_list = prod_bdd_curr_list + self.dfa_bdd_x_list

        sym_lbl_cubes = self._create_lbl_cubes()

        while True: 
            if self.winning_states[layer].compare(self.winning_states[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                init_state_cube = list(((self.init_TS & self.init_DFA) & self.winning_states[layer]).generate_cubes())[0]
                init_val = init_state_cube[1]
                if init_val != math.inf:
                    print(f"A Winning Strategy Exists!!. The Min Energy is {init_val}")
                    return strategy
                else:
                    print("No Winning Strategy Exists!!!")
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
                    # if curr_act_name == 'release' or curr_act_name == 'transit b0':
                    #     print('Wait!')
                    pre_states: BDD = self.get_pre_states(ts_action=tr_action, From=succ_states, prod_curr_list=prod_bdd_curr_list, act_name=curr_act_name)

                    _pre_buckets[act_val + sval] |= pre_states.toADD()
            
            # unions of all predecessors
            pre_states: ADD = reduce(lambda x, y: x | y, _pre_buckets.values())

            # now take univ abstraction to remove edges to states with infinity value
            upre_states: ADD = pre_states.univAbstract(self.env_cube)

            # print non-zero states in this iteration
            if verbose:
                print(f"Non-Zero states at Iteration {layer + 1}")
                self.get_prod_states_from_dd(dd_func=upre_states.existAbstract(self.sys_cube), sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_dfa_bdd_curr_list)
            
            # sys.exit(-1)
            # convert to ADD so that we take max of (s, a_s, a_e)
            tmp_strategy = upre_states.ite(self.manager.addOne(), self.manager.plusInfinity())

            # accomodate for worst-case human behavior
            for sval, apre_s in _pre_buckets.items():
                tmp_strategy = tmp_strategy.max(apre_s.ite(self.manager.addConst(int(sval)), self.manager.addZero()))
            
            # compute the minimum of state action pairs

            # check if this can be done
            strategy = strategy.min(tmp_strategy)

            for tr_dd in self.ts_sym_to_robot_act_map.keys():
                # remove the dependency for that action and preserve the minimum value for every state
                self.winning_states[layer + 1] |= self.winning_states[layer].min(strategy.restrict(tr_dd))
            
            if verbose:
                print(f"Minimum State value at Iteration {layer +1}")
                self.get_state_value_from_dd(dd_func=self.winning_states[layer + 1], sym_lbl_cubes=sym_lbl_cubes, prod_list=[*prod_curr_list, *self.dfa_x_list])
            

            # update counter 
            layer += 1

            sys.exit(-1)
    

    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to roll out the synthesized Min-Max strategy
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

        while True:
            if len(list((self.target_DFA & curr_prod_state).generate_cubes())) > 0:
                target_state_val = list((self.target_DFA & curr_prod_state).generate_cubes())[0][1]
                if target_state_val != math.inf:
                    return
            
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

            list_act_cube = self.convert_add_cube_to_func(act_cube, curr_state_list=self.sys_act_vars)

            # if multiple winning actions exisit from same state
            if len(list_act_cube) > 1:
                ract_name = None
                while ract_name is None:
                    ract_dd: List[int] = random.choice(list_act_cube)
                    ract_name = self.ts_sym_to_robot_act_map.get(ract_dd, None)

            else:
                ract_name = self.ts_sym_to_robot_act_map[act_cube]

            if verbose:
                print(f"Step {counter}: Conf: {self.ts_handle.get_state_from_tuple(curr_ts_tuple)} Act: {ract_name}")
            
            # look up the next tuple 
            next_exp_states = self.ts_handle.adj_map[curr_ts_tuple][ract_name]['r']
            next_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)

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
