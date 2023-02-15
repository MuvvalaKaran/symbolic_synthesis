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
                 dfa_handle: ADDPartitionedDFA,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 int_weight_dict: Dict[str, int],
                 max_ts_action_cost: int,
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
        self.dfa_handle: ADDPartitionedDFA = dfa_handle

        self.int_weight_dict = int_weight_dict

        self.sym_vars_ults: List[ADD] = state_utls_vars
        self.energy_budget: int = budget

        self.sym_trap_state_lbl: ADD = None

        self.max_ts_action_cost: int = max_ts_action_cost

        # index to determine where the state vars start 
        self.state_start_idx: int =  len(self.sym_vars_human) + len(self.sym_vars_robot) + len(dfa_state_vars) + sum([len(listElem) for listElem in self.sym_vars_lbl]) +  len(self.sym_vars_curr)

        self.predicate_sym_map_utls: bidict = {}
        self.utls_cube = reduce(lambda x, y: x & y, self.sym_vars_ults)

        self.prod_adj_map = defaultdict(lambda: defaultdict(lambda: {}))

        num_of_acts = len(list(self.tr_action_idx_map.keys()))
        self.sym_tr_actions = [[self.manager.addZero() for _ in range(len(state_utls_vars))] for _ in range(num_of_acts)]

        # need these two during dfa image computation 
        self.dfa_bdd_x_list: List[BDD] = [i.bddPattern() for i in dfa_state_vars]
        self.dfa_bdd_transition_fun_list: List[BDD] = [i.bddPattern() for i in self.dfa_handle.tr_state_adds]

        # create state util look up dictionary
        self._initialize_add_for_state_utls()
        self.open_list: dict = defaultdict(lambda: set())
        self.closed: ADD = self.manager.addZero()
        self.state_action_bdd: BDD = self.manager.bddZero()

        # count # of state in this graph
        self.scount: int = 0

        # self.leaf_nodes: ADD = self.manager.plusInfinity()
        self.leaf_node_list = defaultdict(lambda: self.manager.addZero())
        self.leaf_vals = set()
        self.lcount: int = 0

        # best alternative ADD and set
        self.ba_strategy: ADD = self.manager.plusInfinity()
        self.ba_set = set()
    

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
    

    def get_trap_state_lbl(self, boxes) -> ADD:
        """
         A helper function that constructs the label asscoatied with trap state and return the ADD. 

         The label looks like (on b1 empty), (on b2 empty), ... 
        """
        # create the corresponding preds
        var_preds = [self.predicate_sym_map_lbl[f'(on {b} empty)'] for b in boxes]

        trap_lbl = reduce(lambda x, y: x & y, var_preds)
        assert not (trap_lbl).isZero(), "Error computing lbl associated with Trap state. It should not be Flase. Fix This!!!"
        return trap_lbl
    

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
    

    def get_dfa_evolution(self, next_sym_state: ADD, curr_dfa_sym: ADD) -> ADD:
        """
         A helper function that given the evolution on the TS (next state), checks if it satisfies any of the DFA edges or not.
          If yes, return the new DFA state.
        """
        # convert the 0-1 ADD to BDD for DFA edge checking
        curr_ts_lbl: BDD = next_sym_state.existAbstract(self.state_cube & self.utls_cube).bddPattern()

        # create DFA edge and check if it satisfies any of the edges or not
        for dfa_state in self.dfa_handle.dfa_predicate_add_sym_map_curr.values():
            bdd_dfa_state: BDD = dfa_state.bddPattern()
            dfa_pre: BDD = bdd_dfa_state.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
            edge_exists: bool = not (dfa_pre & (curr_dfa_sym.bddPattern() & curr_ts_lbl)).isZero()

            if edge_exists:
                curr_dfa_sym: ADD = dfa_state
                break
        
        return curr_dfa_sym, self.dfa_handle.dfa_predicate_add_sym_map_curr.inv[curr_dfa_sym]
    

    def compute_graph_of_utility_reachable_states(self, mod_act_dict: dict, boxes: List[str], verbose: bool = False):
        """
        A function to construct the graph of utility given an Edge Weighted Arena (EWA).
            
        We populate all the states with all the possible utilities. A position may be reachable by several paths,
         therefore it will be duplicated as many times as there are different path utilities.
         This duplication is bounded the value B = 2 * W * |S|. Refer to Lemma 4 of the paper for more details.

        Constructing G' (Graph of utility (TWA)):
        
        S' = S x [B]; Where B is the an positive integer defined as above
        An edge between two states (s, u),(s, u') exists iff s to s' is a valid edge in G and u' = u + w(s, s')
        
        C' = S' ∩ [C1 x [B]] are the target states in G'. The edge weight of edges transiting to a target state (s, u)
        is u. All the other edges have an edge weight 0. (Remember this is a TWA with non-zero edge weights on edges
        transiting to the target states.)
        """
        init_state_tuple = self.get_tuple_from_state(self.init)
        init_dfa_state = (self.dfa_handle.init[0])

        self.sym_trap_state_lbl: ADD = self.get_trap_state_lbl(boxes)

        layer = 0

        self.open_list[layer].add((init_state_tuple, init_dfa_state))

        prod_curr_list: List[ADD] = [*self.dfa_handle.sym_add_vars_curr]
        prod_curr_list.extend([lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list])
        prod_curr_list.extend([*self.sym_vars_curr, *self.sym_vars_ults])

        # used to break the loop
        empty_bucket_counter: int = 0
        
        while True:

            if len(self.open_list[layer]) > 0:
                # if verbose:
                print(f"********************Layer: {layer}**************************")
                
                # reset the empty bucket counter 
                empty_bucket_counter = 0

                # loop over all the valid actions from each state from the current layer, 
                for curr_prod_tuple in self.open_list[layer]:
                    curr_state_tuple = curr_prod_tuple[0]
                    curr_dfa_tuple = curr_prod_tuple[1]
                    
                    # get the sym repr of the DFA state
                    curr_dfa_sym_state: ADD = self.dfa_handle.dfa_predicate_add_sym_map_curr[curr_dfa_tuple]
                    
                    # get sym repr of current and next state
                    curr_ts_sym_state: ADD = self.get_sym_state_from_tuple(curr_state_tuple)
                    
                    # create current sym prod state
                    curr_prod_sym_state: ADD = curr_ts_sym_state & curr_dfa_sym_state

                    # if the current prod state is an accepting state then add it to the leaf ADD along with the state value 
                    if not (curr_dfa_sym_state & self.dfa_handle.sym_goal_state).isZero():
                        if verbose:
                            curr_ts_exp_states = self.get_state_from_tuple(curr_state_tuple) 
                            print(f"Adding leaf node ({curr_ts_exp_states}, {curr_dfa_tuple}) with value {layer}")
                        
                        self.leaf_node_list[layer] |= curr_prod_sym_state & self.predicate_sym_map_utls[layer]
                        self.leaf_vals.add(layer)
                        # update counter
                        self.lcount += 1
                        self.closed |= curr_prod_sym_state & self.predicate_sym_map_utls[layer]
                        continue
                        
                    # update the closed set
                    assert (self.closed & curr_prod_sym_state & self.predicate_sym_map_utls[layer]).isZero(), "Error unrolling the graph. Encountered a twice during unrolling. FIX THIS!!!"
                    self.closed |= curr_prod_sym_state & self.predicate_sym_map_utls[layer]

                    self.scount += 1

                    # loop over all valid action
                    for robot_act in self.ts_handle.org_adj_map[curr_state_tuple].keys():
                        # get the utility of the successor state
                        succ_utl: int = layer + self.int_weight_dict[robot_act]

                        # get the modified robot action name
                        mod_raction_name: str = mod_act_dict[robot_act] 

                        # due to the Value Iteration algo. implementation for purely symbolic Graph of utility,
                        # we need to create a BDD that keeps track of valid state action pairs on this graph.
                        # add (s, a_s) to set of valid state robot-action pairs
                        self.state_action_bdd |= (curr_prod_sym_state & self.predicate_sym_map_utls[layer] & self.ts_handle.predicate_sym_map_robot[mod_raction_name]).bddPattern()

                        if succ_utl <= self.energy_budget:

                            for next_act, next_exp_state in self.ts_handle.org_adj_map[curr_state_tuple][robot_act].items():
                                # first loop over all the human intervening move and then the non-intervening one
                                next_sym_state: ADD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)
                                next_state_tuple = self.get_tuple_from_state(next_exp_state) 

                                # get the DFA state
                                _, next_dfa_tuple = self.get_dfa_evolution(next_sym_state=next_sym_state, curr_dfa_sym=curr_dfa_sym_state)
                                
                                if verbose:
                                    curr_ts_exp_states = self.get_state_from_tuple(curr_state_tuple)
                                    print(f"Adding Human edge: ({curr_ts_exp_states}, {curr_dfa_tuple})[{layer}] -------{robot_act}{next_act}------> ({next_exp_state},{next_dfa_tuple})[{succ_utl}]")
                                
                                # build the adj map needed durong graph of best response construction
                                if 'human' in next_act:
                                    assert self.prod_adj_map.get(curr_prod_tuple, {}).get(robot_act, {}).get(next_act) is None, "Error Computing Adj Dictionary, Fix this!!!"
                                    self.prod_adj_map[(*curr_prod_tuple, layer)][robot_act][next_act] = (next_state_tuple, next_dfa_tuple, succ_utl)
                                else:
                                    assert self.prod_adj_map.get(curr_prod_tuple, {}).get(robot_act, {}).get('r') is None, "Error Computing Adj Dictionary, Fix this!!!"
                                    self.prod_adj_map[(*curr_prod_tuple, layer)][robot_act]['r'] = (next_state_tuple, next_dfa_tuple, succ_utl)

                                # add them to their respective bucket. . .
                                self.open_list[succ_utl].add((next_state_tuple, next_dfa_tuple))
                        
                        else:
                            next_state_tuple: tuple = (self.pred_int_map['(trap-state)'])
                            # update the adj map tp the trap state
                            assert self.prod_adj_map.get(curr_prod_tuple, {}).get(robot_act, {}).get('r') is None, "Error Computing Adj Dictionary, Fix this!!!"
                            self.prod_adj_map[(*curr_prod_tuple, layer)][robot_act]['r'] = (next_state_tuple, curr_dfa_tuple, self.energy_budget)

            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == self.max_ts_action_cost:
                    print(f"Done Computing the Graph of Utility! Accepting Leaf nodes {self.lcount}; Total states {self.scount}")
                    break
            
            layer += 1
    

    def get_best_alternatives(self, cooperative_vals: ADD,  mod_act_dict: dict, verbose: bool = False):
        """
         A function that computes the best alternative from each valid edge in the Graph of Utility.  

         ba(s, s') is defined as minimum of all the cooperative values for successors s'' s.t. s'' not equal to s'.
         If there is no alternate edge to choose from, then ba(s, s') = +inf.
         
         The cooperate values are stored in the winning states ADD along with their optimal values.
        """

        # loop through the open_list dict computed above
        layer = 0
        while True:
            if len(self.open_list[layer]) > 0:
                # if verbose:
                print(f"********************Layer: {layer}**************************")
                
                # reset the empty bucket counter 
                empty_bucket_counter = 0

                for curr_prod_tuple in self.open_list[layer]:
                    curr_state_tuple = curr_prod_tuple[0]
                    curr_dfa_tuple = curr_prod_tuple[1]

                    # get the sym repr of the DFA state
                    curr_dfa_sym_state: ADD = self.dfa_handle.dfa_predicate_add_sym_map_curr[curr_dfa_tuple]

                    # Do not compute best alternative from an accepting state(leaf node in Graph of utility)
                    if not (curr_dfa_sym_state & self.dfa_handle.sym_goal_state).isZero():
                        continue
                        
                    # get sym repr of current and next state
                    curr_ts_sym_state: ADD = self.get_sym_state_from_tuple(curr_state_tuple)

                    # construct the prod state (s, z, u)
                    curr_prod_sym_state: ADD = curr_ts_sym_state & curr_dfa_sym_state & self.predicate_sym_map_utls[layer]

                    # if alternate edge exists then loop over all valid robot actions
                    if len(self.ts_handle.org_adj_map[curr_state_tuple].keys()) > 1:
                        for robot_act_name in self.ts_handle.org_adj_map[curr_state_tuple].keys():
                            # get the modified robot action name
                            mod_raction_name: str = mod_act_dict[robot_act_name]

                            # 0-1 ADD
                            curr_state_act: ADD = curr_prod_sym_state & self.ts_handle.predicate_sym_map_robot[mod_raction_name]

                            bdd_curr_state_act: BDD = curr_state_act.bddPattern()

                            # get the alt edges
                            alt_edges: BDD = self.state_action_bdd & ~bdd_curr_state_act
                            # alt_edges: BDD = coop_bdd_vals & ~bdd_curr_state_act

                            # get all the alternate edges (s, (a_s)') from the current state 
                            alt_edges_state: ADD = alt_edges.toADD() & curr_prod_sym_state
                            alt_edges_state =  alt_edges_state.ite(self.manager.addOne(), self.manager.plusInfinity())

                            min_val: ADD = (alt_edges_state & cooperative_vals).findMin()

                            assert min_val != self.manager.addZero(), "Error computing best alternative value. This should never be zero. Fix this!!!"

                            if min_val == self.manager.plusInfinity():
                                min_val = self.manager.addConst(self.energy_budget)
                            
                            self.ba_strategy = self.ba_strategy.min( (curr_state_act).ite(min_val, self.manager.plusInfinity()))

                            ## add the ba value to set
                            int_min_val: int =  list(min_val.generate_cubes())[0][1]
                            self.ba_set.add(int_min_val)

                            if verbose:
                                curr_ts_exp_states = self.get_state_from_tuple(curr_state_tuple)
                                print(f"******************** Best Alternate Val ({curr_ts_exp_states}, {curr_dfa_tuple})[{layer}] ---{robot_act_name}--->: [{int_min_val}] ")

            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == self.max_ts_action_cost:
                    break
            
            layer += 1
        

        # sanity check. The set of best alternative values should be subset of utility values
        assert self.ba_set.issubset(self.leaf_vals), "Error computing set of beat alternatives. The BA set should be a subset of utility values "

        # add infinity to the set of best responses
        self.ba_set.add(inf)
        
