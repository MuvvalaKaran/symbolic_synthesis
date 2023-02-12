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

from src.symbolic_graphs import ADDPartitionedDFA
from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs
from src.symbolic_graphs.hybrid_regret_graphs import HybridGraphOfUtility
from src.symbolic_graphs.symbolic_regret_graphs import SymbolicGraphOfUtility

from src.algorithms.base import BaseSymbolicSearch
from src.algorithms.strategy_synthesis import AdversarialGame


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


class GraphOfUtlCooperativeGame(BaseSymbolicSearch):
    """
     This class plays a Min-Min game over the graph of utility. As the Graph of utility represents a prooduct (with TS and DFA variable) graph with 
      TR encoding the evolution over the TS and the DFA, we reimplement the Cooperative Game class.
     
      Nonetheless, the Algorithm remains the same. 
    """

    def __init__(self,
                 prod_handle: HybridGraphOfUtility,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 ts_curr_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 ts_utls_vars: List[ADD],
                 sys_act_vars: List[ADD],
                 env_act_vars: List[ADD],
                 cudd_manager: Cudd) -> None:
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.init_prod = self.init_TS & self.init_DFA & prod_handle.predicate_sym_map_utls[0]

        self.ts_x_list = ts_curr_vars
        self.dfa_x_list = dfa_curr_vars

        self.sys_act_vars = sys_act_vars
        self.env_act_vars = env_act_vars

        self.ts_utls_list = ts_utls_vars

        self.prod_trans_func_list: List[List[ADD]] = prod_handle.sym_tr_actions
        self.prod_bdd_trans_func_list: List[List[BDD]] = []

        # need this during preimage computation 
        self.dfa_bdd_x_list = [i.bddPattern() for i in dfa_curr_vars]

        self.ts_action_idx_map: bidict = ts_handle.tr_action_idx_map

        self.ts_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        self.dfa_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_add_sym_map_curr.inv

        self.ts_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_sym_to_robot_act_map: bidict = ts_handle.predicate_sym_map_robot.inv
        
        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle
        self.prod_handle = prod_handle
        
        # create corresponding cubes to avoid repetition
        self.ts_xcube: ADD = reduce(lambda x, y: x & y, self.ts_x_list)
        self.dfa_xcube: ADD = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.ts_obs_cube: ADD = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])
        self.ts_utls_cube: ADD = reduce(lambda x, y: x & y, self.ts_utls_list)
        self.sys_cube: ADD = reduce(lambda x, y: x & y, self.sys_act_vars)
        self.env_cube: ADD = reduce(lambda x, y: x & y, self.env_act_vars)

        # create ts and dfa combines cube
        self.prod_xlist: list =  self.dfa_x_list + [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list] + self.ts_x_list + self.ts_utls_list
        self.prod_xcube: ADD = reduce(lambda x, y: x & y, self.prod_xlist)

        # get the bdd variant
        self.prod_bdd_curr_list = [_avar.bddPattern() for _avar in self.prod_xlist]

        # sys and env cube
        self.sys_env_cube = reduce(lambda x, y: x & y, [*self.sys_act_vars, *self.env_act_vars])

        # ADD that keeps track of the optimal values of state at each iteration
        self.winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())
        
        # get the bdd version of the transition function as vectorComposition only works with
        for act in self.prod_trans_func_list:
            act_ls = []
            for avar in act:
                act_ls.append(avar.bddPattern())
        
            self.prod_bdd_trans_func_list.append(act_ls)
        
        # mimimum energy required from the init states
        self.init_state_value: Union[int, float] = math.inf

    
    def _create_lbl_cubes(self) -> List[ADD]:
        """
        A helper function that creates cubes of each lbl and store them in a list in the same order as the original order.
         These cubes are used when we convert a BDD to lbl state where we need to extract each lbl.
        """
        sym_lbl_xcube_list = [] 
        for vars_list in self.ts_obs_list:
            sym_lbl_xcube_list.append(reduce(lambda x, y: x & y, vars_list))
        
        return sym_lbl_xcube_list


    def get_state_value_from_dd(self, dd_func: ADD, **kwargs) -> None:
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
                if var == 2 and self.manager.addVar(_idx) not in self.prod_xlist:
                    continue
                
                elif self.manager.addVar(_idx) in self.prod_xlist:
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
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(cube.existAbstract(self.dfa_xcube & self.ts_utls_cube), kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"
            _ts_utl: int = self.prod_handle.predicate_sym_map_utls.inv[cube.existAbstract(self.ts_xcube & self.ts_obs_cube & self.dfa_xcube)]

            _dfa_name = self._look_up_dfa_name(prod_dd=cube.existAbstract(self.ts_utls_cube),
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"[({_ts_name}, {_dfa_name}), {_ts_utl}] {val}")

    
    def get_pre_states(self, From: BDD) -> BDD:
        """
         Compute the pre-image on the product graph
        """
        pre_prod_state: BDD = self.manager.bddZero()
        
        for ts_transition in self.prod_bdd_trans_func_list:
            pre_prod_state |= From.vectorCompose(self.prod_bdd_curr_list, [*ts_transition])
            
        return pre_prod_state

    
    def solve(self, verbose: bool = False) -> ADD:
        """
         Method to compute optimal strategies for the robot assuming human to Cooperative for given prod graph
        """
        accp_states = self.prod_handle.leaf_nodes

        # strategy - optimal (state & robot-action) pair stored in the ADD
        strategy: ADD  = self.manager.plusInfinity()

        # initializes accepting states to be zero
        self.winning_states[0] |= self.winning_states[0].min(accp_states)
        strategy = strategy.min(accp_states)

        layer: int = 0

        sym_lbl_cubes = self._create_lbl_cubes()

        while True:
            if self.winning_states[layer].compare(self.winning_states[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                init_state_cube = list((self.init_prod & self.winning_states[layer]).generate_cubes())[0]
                init_val: int = init_state_cube[1]
                self.init_state_value = init_val
                if init_val != math.inf:
                    print(f"A Winning Strategy Exists!!. The Min Energy is {init_val}")
                    return strategy
                else:
                    print("No Winning Strategy Exists!!!")
                    # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
                    del self.winning_states
                    return
            
            print(f"**************************Layer: {layer}**************************")

            _win_state_bucket: Dict[BDD] = defaultdict(lambda: self.manager.bddZero())
            
            # convert the winning states into buckets of BDD
            for sval in self.prod_handle.leaf_vals:
                # get the states with state value equal to sval and store them in their respective bukcets
                win_sval = self.winning_states[layer].bddInterval(sval, sval)

                if not win_sval.isZero():
                    _win_state_bucket[sval] |= win_sval
            
            _pre_buckets: Dict[ADD] = defaultdict(lambda: self.manager.addZero())

            # compute the predecessor and store them by successor cost
            for sval, succ_states in _win_state_bucket.items():
                pre_states: BDD = self.get_pre_states(From=succ_states)

                if not pre_states.isZero():
                    _pre_buckets[sval] |= pre_states.toADD()
            
            # unions of all predecessors
            pre_states: ADD = reduce(lambda x, y: x | y, _pre_buckets.values())

            tmp_strategy: ADD = pre_states.ite(self.manager.addOne(), self.manager.plusInfinity())

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
                self.get_state_value_from_dd(dd_func=self.winning_states[layer + 1], sym_lbl_cubes=sym_lbl_cubes)

            # update counter 
            layer += 1
    
    
    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        raise NotImplementedError()



class SymbolicGraphOfUtlCooperativeGame(CooperativeGame):

    def __init__(self,
                 gou_handle: SymbolicGraphOfUtility,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 ts_curr_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 ts_utls_vars: List[ADD],
                 sys_act_vars: List[ADD],
                 env_act_vars: List[ADD],
                 cudd_manager: Cudd) -> None:
        super().__init__(ts_handle=ts_handle,
                         dfa_handle=dfa_handle,
                         ts_curr_vars=ts_curr_vars,
                         dfa_curr_vars=dfa_curr_vars,
                         sys_act_vars=sys_act_vars,
                         env_act_vars=env_act_vars,
                         ts_obs_vars=ts_obs_vars,
                         cudd_manager=cudd_manager)


        self.init_prod = self.init_TS & self.init_DFA & gou_handle.predicate_sym_map_utls[0]

        self.ts_utls_list = ts_utls_vars

        self.utls_trans_func_list: List[List[ADD]] = gou_handle.sym_tr_actions
        self.ults_bdd_trans_func_list: List[List[BDD]] = []

        self.gou_handle = gou_handle
        
        # create corresponding cubes to avoid repetition
        self.ts_utls_cube: ADD = reduce(lambda x, y: x & y, self.ts_utls_list)

        # energy budget
        self.energy_budget: int = gou_handle.energy_budget
      

        # create ts and dfa combines cube
        self.prod_xlist: list =  self.dfa_x_list + [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list] + self.ts_x_list + self.ts_utls_list
        self.prod_xcube: ADD = reduce(lambda x, y: x & y, self.prod_xlist)

        # # get the bdd variant
        # self.prod_bdd_curr_list = [_avar.bddPattern() for _avar in self.prod_xlist]
        
        # get the bdd version of the transition function as vectorComposition only works with
        for act in self.utls_trans_func_list:
            act_ls = []
            for avar in act:
                act_ls.append(avar.bddPattern())
        
            self.ults_bdd_trans_func_list.append(act_ls)
    

    def get_state_value_from_dd(self, dd_func: ADD, **kwargs) -> None:
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
                if var == 2 and self.manager.addVar(_idx) not in self.prod_xlist:
                    continue
                
                elif self.manager.addVar(_idx) in self.prod_xlist:
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
            # convert to 0-1 ADD
            ts_cube = cube.existAbstract(self.dfa_xcube & self.ts_utls_cube) #.bddPattern().toADD()
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(ts_cube, kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _ts_utl: int = self.gou_handle.predicate_sym_map_utls.inv[cube.existAbstract(self.ts_xcube & self.ts_obs_cube & self.dfa_xcube)] #.bddPattern().toADD()]

            _dfa_name = self._look_up_dfa_name(prod_dd=cube.existAbstract(self.ts_utls_cube), #.bddPattern().toADD(),
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"[({_ts_name}, {_dfa_name}), {_ts_utl}] {val}")


    def get_pre_states(self, ts_action: List[BDD], From: BDD, prod_curr_list=None, **kwargs) -> BDD:
        """
         Compute the pre-image on the product graph
        """
        ults_tr = kwargs['utls_tr']
        # # first evolve over DFA and then evolve over the TS and utility values
        # mod_win_state: BDD = From.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
        
        pre_prod_state: BDD = From.vectorCompose(prod_curr_list, [*ts_action, *ults_tr])
            
        return pre_prod_state
    

    def solve(self, verbose: bool = False) -> ADD:
        """
         Compute cVals on the Symbolic Graph of utility.
        """
        
        ts_states: ADD = self.obs_add
        accp_states: ADD = ts_states & self.target_DFA

        # convert it to 0 - Infinity ADD
        # accp_states = accp_states.ite(self.manager.addZero(), self.manager.plusInfinity())

        # strategy - optimal (state & robot-action) pair stored in the ADD
        strategy: ADD  = self.manager.plusInfinity()

        # initializes accepting states
        for sval in range(self.energy_budget + 1):
            # augment the accepting states with utility vars
            tmp_accp_states = accp_states & self. gou_handle.predicate_sym_map_utls[sval]
            tmp_accp_states = tmp_accp_states.ite(self.manager.addConst(int(sval)), self.manager.plusInfinity())
            self.winning_states[0] |= self.winning_states[0].min(tmp_accp_states)
        
            strategy |= strategy.min(tmp_accp_states)

        layer: int = 0

        sym_lbl_cubes = self._create_lbl_cubes()

        prod_curr_list = []
        prod_curr_list.extend([lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])
        prod_curr_list.extend(self.ts_x_list)
        prod_curr_list.extend(self.ts_utls_list)
        
        prod_bdd_curr_list = [_avar.bddPattern() for _avar in prod_curr_list]

        while True:
            if self.winning_states[layer].compare(self.winning_states[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                init_state_cube = list((self.init_prod & self.winning_states[layer]).generate_cubes())[0]
                init_val: int = init_state_cube[1]
                self.init_state_value = init_val
                if init_val != math.inf:
                    print(f"A Winning Strategy Exists!!. The Min Energy is {init_val}")
                    return strategy
                else:
                    print("No Winning Strategy Exists!!!")
                    # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
                    del self.winning_states
                    return
            

            print(f"**************************Layer: {layer}**************************")

            _win_state_bucket: Dict[BDD] = defaultdict(lambda: self.manager.bddZero())
            
            # convert the winning states into buckets of BDD
            for sval in range(self.gou_handle.energy_budget + 1):
                # get the states with state value equal to sval and store them in their respective bukcets
                win_sval = self.winning_states[layer].bddInterval(sval, sval)

                if not win_sval.isZero():
                    _win_state_bucket[sval] |= win_sval
            

            _pre_buckets: Dict[ADD] = defaultdict(lambda: self.manager.addZero())

            # compute the predecessor and store them by successor cost
            for sval, succ_states in _win_state_bucket.items():
                # first evolve over DFA and then evolve over the TS and utility values
                mod_win_state: BDD = succ_states.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
                for tr_action, utls_tr in zip(self.ts_bdd_transition_fun_list, self.ults_bdd_trans_func_list):
                    pre_states: BDD = self.get_pre_states(ts_action=tr_action, From=mod_win_state, prod_curr_list=prod_bdd_curr_list, utls_tr=utls_tr)
    
                    if not pre_states.isZero():
                        _pre_buckets[sval] |= pre_states.toADD()
            
            # unions of all predecessors
            pre_states: ADD = reduce(lambda x, y: x | y, _pre_buckets.values())

            tmp_strategy: ADD = pre_states.ite(self.manager.addOne(), self.manager.plusInfinity())

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
                self.get_state_value_from_dd(dd_func=self.winning_states[layer + 1], sym_lbl_cubes=sym_lbl_cubes)

            # update counter 
            layer += 1
    
    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        raise NotImplementedError()
