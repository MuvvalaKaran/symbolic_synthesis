'''
 This scripts all the functions to construct the graph of best repsonse. 
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
from src.symbolic_graphs import SymbolicGraphOfUtility


class SymbolicGraphOfBR(DynWeightedPartitionedFrankaAbs):


    def __init__(self,
                 curr_vars: list,
                 lbl_vars: list,
                 robot_action_vars: list,
                 human_action_vars: list,
                 task, domain,
                 ts_state_map: dict,
                 ts_states: list,
                 manager: Cudd,
                 weight_dict: dict,
                 seg_actions: dict,
                 ts_state_lbls: list,
                 dfa_state_vars: List[ADD],
                 sup_locs: List[str],
                 top_locs: List[str],
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 symbolic_gou_handle: SymbolicGraphOfUtility,
                 prod_ba_vars: List[ADD],
                 prod_succ_ba_vars: List[ADD],
                 **kwargs):
        super().__init__(curr_vars,
                         lbl_vars,
                         robot_action_vars,
                         human_action_vars,
                         task, domain,
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
        self.gou_handle: SymbolicGraphOfUtility = symbolic_gou_handle

        self.ba_set: set = symbolic_gou_handle.ba_set
        # ba vars for current state
        self.sym_vars_ba: List[ADD] = prod_ba_vars
        
        # ba vars used to augment org robot action with successor ba vars
        self.sym_vars_succ_ba: List[ADD] = prod_succ_ba_vars

        self.predicate_sym_map_ba: bidict = {}

        # create dictionaries to keep track of states as we expand and added to the TR
        self.open_list  = defaultdict(lambda: set())
        self.closed = self.manager.addZero()

        # count # of states, leaf nodes, and edges in this Graph of Best Response
        self.scount: int = 0
        self.lcount: int  = 0

        self.leaf_nodes: ADD = self.manager.plusInfinity()
        self.leaf_vals = set()

        self.prod_adj_map = defaultdict(lambda: defaultdict(lambda: {}))

        # the # of vars in the TS state has increased as we have additional varibale associted with state utility. 
        num_ts_state_vars: int = sum([len(listElem) for listElem in self.sym_vars_lbl]) + len(self.sym_vars_curr) + len(symbolic_gou_handle.sym_vars_ults) + len(self.sym_vars_ba)
        num_dfa_state_vars: int = len(dfa_state_vars)
        num_of_acts = len(list(self.tr_action_idx_map.keys()))
        self.sym_tr_actions = [[self.manager.addZero() for _ in range(num_ts_state_vars + num_dfa_state_vars)] for _ in range(num_of_acts)]

        # index to determine where the state vars start 
        self.state_start_idx: int =  len(self.sym_vars_human) + len(self.sym_vars_robot)

        # initialize mapping from ba val to boolean formula
        self._initialize_adds_for_ba_vars()

    
    def _initialize_adds_for_ba_vars(self):
        """
         This function initializes ADD that represents the Prod state Best response value at each state. 
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_ba)))
        _node_int_map = bidict({ba_val: boolean_str[idx] for idx, ba_val in enumerate(self.ba_set)})

        assert len(boolean_str) >= len(_node_int_map), \
             "FIX THIS: Looks like there are more BA values than it's corresponding boolean variables. Fix this!!!"
        
        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map.items():
            _val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _val_list.append(self.sym_vars_ba[_idx])
                else:
                    _val_list.append(~self.sym_vars_ba[_idx])
            
            _bool_func_curr = reduce(lambda a, b: a & b, _val_list)

            # update bidict accordingly
            _node_int_map[_key] = _bool_func_curr
        
        self.predicate_sym_map_ba = bidict(_node_int_map)
    

    def add_edge_to_trap_state(self,
                               curr_prod_tuple: tuple,
                               curr_prod_sym_state: ADD,
                               mod_act_dict: dict,
                               trap_state_sym: ADD,
                               robot_act_name: str,
                               prod_curr_list: List[ADD],
                               valid_hact_list: ADD = None,
                               debug: bool = True,
                               verbose: bool = False):
        """
         A helper function to add the an edge to the trap state.
        """
        trap_state_tuple: tuple = (self.pred_int_map['(trap-state)'])
        init_dfa_tuple: int = self.dfa_handle.init[0]
        
        
        self.add_edge_to_action_tr(curr_state_tuple=curr_prod_tuple,
                                   curr_state_sym=curr_prod_sym_state,
                                   next_state_tuple=(trap_state_tuple, init_dfa_tuple, self.gou_handle.energy_budget, inf),
                                   mod_act_dict=mod_act_dict,
                                   nxt_state_sym=trap_state_sym,
                                   valid_hact_list=valid_hact_list,
                                   robot_action_name=robot_act_name,
                                   prod_curr_list=prod_curr_list,
                                   debug=debug)
        
        if verbose:
            curr_ts_exp_states: List[str] = self.get_state_from_tuple(curr_prod_tuple[0])
            print(f"Adding Trap edge: " \
                f"({curr_ts_exp_states}, {curr_prod_tuple[1]}, {curr_prod_tuple[2]}, {curr_prod_tuple[3]})" \
                f"-------{robot_act_name}------> (vT)")
        
        


    def add_state_to_leaf_node(self,
                               curr_utls_tuple: int,
                               next_ts_tuple: tuple,
                               next_dfa_tuple: int,
                               next_utls_tuple: int,
                               next_br_tuple: int,
                               next_prod_sym_state: ADD,
                               only_leaf_nodes: bool = False) -> None:
        """
         A helper function that adds an accepting state to the set of leaf nodes.
        """
        # if not  (next_dfa_sym & self.dfa_handle.sym_goal_state).isZero():
        # compute edge weight.
        # accp_w: int =  curr_utls_tuple - min(curr_utls_tuple, next_br_tuple)
        accp_w: int =  next_utls_tuple - min(next_utls_tuple, next_br_tuple)
        
        next_ts_exp_state = self.get_state_from_tuple(next_ts_tuple)
        
        if only_leaf_nodes:
            print(f"Adding leaf node ({next_ts_exp_state}, {next_dfa_tuple}, {next_utls_tuple}, {next_br_tuple}) with edge weight {accp_w}")
        
        # create (state-val) - infinity ADD
        full_prod_state_val: ADD = (next_prod_sym_state).ite(self.manager.addConst(accp_w), self.manager.plusInfinity())

        self.leaf_nodes = self.leaf_nodes.min(full_prod_state_val)
        self.leaf_vals.add(accp_w)

        # update counter
        self.lcount += 1
        self.closed |= next_prod_sym_state



    # add code to construt the edge TR
    def add_edge_to_action_tr(self,
                              curr_state_tuple: tuple,
                              next_state_tuple: tuple,
                              curr_state_sym: ADD,
                              nxt_state_sym: ADD,
                              mod_act_dict: dict,
                              curr_str_state: List[str] = '',
                              next_str_state: List[str] = '',
                              robot_action_name: str = '',
                              human_action_name: str = '',
                              valid_hact_list: ADD = None,
                              **kwargs):
        """
         Create the edge for the Graph of Best Response and store in a Compositional fashion. 
        """
        if valid_hact_list is not None:
            assert isinstance(valid_hact_list, ADD), "Error Constructing TR Edges. Fix This!!!"
            no_human_move = valid_hact_list
        
        if human_action_name != '':
            assert robot_action_name != '', "Error While constructing Human Edge, FIX THIS!!!"
        
        # get the modified robot action name
        mod_raction_name: str = mod_act_dict[robot_action_name]
        robot_move: ADD = self.ts_handle.predicate_sym_map_robot[mod_raction_name]

        _tr_idx: int = self.tr_action_idx_map.get(robot_action_name)

        if human_action_name != '':
            # get the modified human action name
            mod_haction_name: str = mod_act_dict[human_action_name]

            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & robot_move & self.ts_handle.predicate_sym_map_human[mod_haction_name]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_state_tuple} --- {robot_action_name} & {human_action_name}---> {next_state_tuple}")
                
                self.mono_tr_bdd |= curr_state_sym & robot_move & self.ts_handle.predicate_sym_map_human[mod_haction_name]
            
        else:
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & robot_move & no_human_move).isZero()
            
            if not edge_exist:
                print(f"Nondeterminism due to Human Action: {curr_state_tuple} ---{robot_action_name}---> {next_state_tuple}")
            
            self.mono_tr_bdd |= curr_state_sym & robot_move & no_human_move

        # generate all the cubes, with their corresponding string repr and leaf value (state value should be 1)
        add_cube: List[Tuple(list, int)] = list(nxt_state_sym.generate_cubes())   
        assert len(add_cube) == 1, "Error computing cube string for next state's symbolic representation. FIX THIS!!!"
        assert add_cube[0][1] == 1, "Error computing next state cube. The integer value of the leaf node in the ADD is not 1. FIX THIS!!!"

        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(add_cube[0][0]):
            if var == 1 and self.manager.addVar(_idx) in kwargs['prod_curr_list']:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                # if human intervenes then the edge looks like (true) & (human move b# l# l#)
                if human_action_name != '':
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & self.ts_handle.predicate_sym_map_human[mod_haction_name]
                # if human does not intervene then the edge looks like (robot-action) & not(valid human moves)
                else:
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & no_human_move

            
            elif var == 2 and self.manager.addVar(_idx) in kwargs['prod_curr_list']:
                warnings.warn("Encountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)
        
        # create the adj map for rollout purposes
        if human_action_name != '':
            # assert self.prod_adj_map.get(curr_state_tuple, {}).get(robot_action_name, {}).get(human_action_name) is None, "Error Computing Adj Dictionary, Fix this!!!"
            # self.prod_adj_map[curr_state_tuple][robot_action_name][human_action_name] = next_state_tuple
            assert self.prod_adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get(mod_haction_name) is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.prod_adj_map[curr_state_tuple][mod_raction_name][mod_haction_name] = next_state_tuple
        
        else:
            # assert self.prod_adj_map.get(curr_state_tuple, {}).get(robot_action_name, {}).get('r') is None, "Error Computing Adj Dictionary, Fix this!!!"
            # self.prod_adj_map[curr_state_tuple][robot_action_name]['r'] = next_state_tuple
            assert self.prod_adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get('r') is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.prod_adj_map[curr_state_tuple][mod_raction_name]['r'] = next_state_tuple

        # update edge count 
        self.ecount += 1


    def construct_graph_of_best_response(self, mod_act_dict: dict, print_leaf_nodes: bool = True, verbose: bool = False, debug: bool = True):
        """
         Main function to construct the Graph of Best Response. We construct all the possible state (S, B) where S is a state in the product graph
          and B is the Best reponse at the current state. From (S, B), we add an edge to (S', B') iff (S, S') is a valid edge on Graph of Utility and 
          B' is min(B, ba(S, S'). 

          We then add the edge to the TR. All the edges have weight exactly zero while weight to the accepting state (also an leaf node) is defined as 

          w(e_f) := u - min(u, b) for S_f = ((s, u), b). HEre S_f is a accepting state and s is a states in the TS, u \in U is the utlity value on the graph of utility and 
          b \in BA is the best response value. Note e_f is an edge to the accpeting state in the Graph of Best Response. 
        """
        # get the init state in the Grapg of Utility
        init_gou_state_tuple: tuple = next(iter(self.gou_handle.open_list[0]))

        sym_trap_state_lbl: ADD = self.gou_handle.sym_trap_state_lbl
        sym_trap_state: ADD = self.get_sym_state_from_exp_states(exp_states=['(trap-state)'])
        prod_trap_state_sym: ADD = sym_trap_state & sym_trap_state_lbl &  self.dfa_handle.sym_init_state & self.gou_handle.predicate_sym_map_utls[self.gou_handle.energy_budget] & self.predicate_sym_map_ba[inf]

        # Add trap-state to the set of explored states
        self.closed |= prod_trap_state_sym

        layer = 0
        
        # creating monolithinc tr bdd to keep track all the transition we are creating to detect nondeterminism
        if debug:
            self.mono_tr_bdd = self.manager.addZero() 
        
        # if verbose flag is True then print leaf nodes too.
        if verbose:
            print_leaf_nodes= True
        
        # the init state is of the Form ((ts-tuple), DFA, Utl val, BR Value)
        self.open_list[layer].add((*init_gou_state_tuple, 0, inf))

        # prod state have TS, DFA, Utls and BA vars
        prod_curr_list: List[ADD] = [*self.dfa_handle.sym_add_vars_curr]
        prod_curr_list.extend([lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list])
        prod_curr_list.extend([*self.sym_vars_curr, *self.gou_handle.sym_vars_ults, *self.sym_vars_ba])


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
                    curr_ts_tuple = curr_prod_tuple[0]
                    curr_dfa_tuple = curr_prod_tuple[1]
                    curr_utls_tuple = curr_prod_tuple[2]
                    curr_br_tuple = curr_prod_tuple[3]

                    # get the sym repr of the DFA state
                    curr_dfa_sym_state: ADD = self.dfa_handle.dfa_predicate_add_sym_map_curr[curr_dfa_tuple]

                    # get the sym repr of the utls val
                    curr_utls_sym : ADD = self.gou_handle.predicate_sym_map_utls[curr_utls_tuple]

                    # get the sym repr of the br val
                    curr_ba_sym : ADD = self.predicate_sym_map_ba[curr_br_tuple]

                    # get sym repr of current and next state
                    curr_ts_sym_state: ADD = self.get_sym_state_from_tuple(curr_ts_tuple)
                    curr_ts_exp_states = self.get_state_from_tuple(curr_ts_tuple)

                    # create current sym prod state
                    curr_prod_sym_state: ADD = curr_ts_sym_state & curr_dfa_sym_state & curr_utls_sym & curr_ba_sym

                    self.scount += 1

                    # for every valus successors state in the Graph of Utility. . 
                    for robot_act in self.gou_handle.prod_adj_map[(curr_ts_tuple, curr_dfa_tuple, curr_utls_tuple)]:
                        # edge corresponding to human intervention
                        no_human_move_edge: ADD = self.manager.addOne()
                        
                        valid_act_list = []

                        mod_raction_name: str = mod_act_dict[robot_act]
                        robot_move: ADD = self.ts_handle.predicate_sym_map_robot[mod_raction_name]

                        # get the corresponding ba values
                        ba_sym: ADD = curr_ts_sym_state & curr_dfa_sym_state & curr_utls_sym & robot_move & self.gou_handle.ba_strategy

                        ba_cube: List[Tuple(list, int)] = list(ba_sym.generate_cubes())   
                        assert len(ba_cube) == 1, "Error computing cube string for the symbolic representation of the next state's best alternate response. FIX THIS!!!"

                        next_ba_int = min(ba_cube[0][1], curr_br_tuple)
                        next_ba_sym = self.predicate_sym_map_ba[next_ba_int]

                        for human_act, next_state_tuple in self.gou_handle.prod_adj_map[(curr_ts_tuple, curr_dfa_tuple, curr_utls_tuple)][robot_act].items():
                            # first loop over all the human intervening move and then the non-intervening one
                            # Under human action, we can only transit from (S, B) -----> (S', B') where B' = B

                            if human_act == 'r':
                                continue

                            # update list of valid human actions
                            valid_act_list.append(human_act)

                            # parse the next state tuple
                            next_ts_tuple = next_state_tuple[0]
                            next_dfa_tuple = next_state_tuple[1]
                            next_utls_tuple = next_state_tuple[2]

                            if len(next_ts_tuple) == 1 and 'trap' in self.pred_int_map[next_ts_tuple]:
                                # adding edge to the trap state
                                warnings.warn("We should not have encountered an edge to the Trap state under human action. Error in Graph of Utility construction. Fix This!!!")
                                sys.exit(-1)

                            # get successor TS state tuple and sym repr
                            next_sym_state: ADD = self.get_sym_state_from_tuple(next_ts_tuple)

                            # successor sym DFA repr
                            next_dfa_sym: ADD = self.dfa_handle.dfa_predicate_add_sym_map_curr[next_dfa_tuple]

                            # successor sym utls repr
                            next_utls_sym : ADD = self.gou_handle.predicate_sym_map_utls[next_utls_tuple]

                            # compute the next prod state
                            next_prod_sym_state: ADD = next_sym_state & next_dfa_sym & next_utls_sym & next_ba_sym

                            self.add_edge_to_action_tr(curr_state_tuple=curr_prod_tuple,
                                                       curr_state_sym=curr_prod_sym_state,
                                                       next_state_tuple=(*next_state_tuple, next_ba_int),
                                                       mod_act_dict=mod_act_dict,
                                                       nxt_state_sym=next_prod_sym_state,
                                                       human_action_name=human_act,
                                                       robot_action_name=robot_act,
                                                       prod_curr_list=prod_curr_list,
                                                       debug=debug)
                            if verbose:
                                next_ts_exp_state = self.get_state_from_tuple(next_ts_tuple)
                                print(f"Adding Human edge: " \
                                     f"({curr_ts_exp_states}, {curr_dfa_tuple}, {curr_utls_tuple}, {curr_br_tuple})" \
                                     f"-------{robot_act} {human_act}------> ({next_ts_exp_state}, {next_dfa_tuple}, {next_utls_tuple}, {next_ba_int})")
                            
                            
                            # if the successor prod state is an accepting state then add it to the leaf ADD along with the state value as described in the docstring
                            if not (next_dfa_sym & self.dfa_handle.sym_goal_state).isZero():
                                self.add_state_to_leaf_node(curr_utls_tuple=curr_utls_tuple,
                                                            next_ts_tuple=next_ts_tuple,
                                                            next_dfa_tuple=next_dfa_tuple,
                                                            next_utls_tuple=next_utls_tuple,
                                                            next_br_tuple=next_ba_int,
                                                            next_prod_sym_state=next_prod_sym_state,
                                                            only_leaf_nodes=print_leaf_nodes)
                                self.closed |= next_prod_sym_state
                                continue

                            
                            # add them to their respective bucket. . .
                            self.open_list[next_utls_tuple].add((*next_state_tuple, next_ba_int))
                            self.closed |= next_prod_sym_state
                        
                        # now add robot edges with no human-intervention
                        # if there are any valid human edges from curr state
                        if len(valid_act_list) > 0:
                            valid_hact: List[ADD] = [self.ts_handle.predicate_sym_map_human[mod_act_dict[ha]] for ha in valid_act_list]
                            no_human_move_edge: ADD = ~(reduce(lambda x, y: x | y, valid_hact))    

                            assert not no_human_move_edge.isZero(), "Error computing a human no-intervene edge. FIX THIS!!!"
                        
                        
                        next_state_tuple = self.gou_handle.prod_adj_map[(curr_ts_tuple, curr_dfa_tuple, curr_utls_tuple)][robot_act]['r']
                        next_ts_tuple = next_state_tuple[0]
                        next_dfa_tuple = next_state_tuple[1]
                        next_utls_tuple = next_state_tuple[2]

                        if isinstance(next_ts_tuple, int) and 'trap' in self.pred_int_map.inv[next_ts_tuple]:
                            # adding edge to the trap state
                            self.add_edge_to_trap_state(curr_prod_tuple=curr_prod_tuple,
                                                        curr_prod_sym_state=curr_prod_sym_state,
                                                        mod_act_dict=mod_act_dict,
                                                        trap_state_sym=prod_trap_state_sym,
                                                        robot_act_name=robot_act,
                                                        valid_hact_list=no_human_move_edge,
                                                        prod_curr_list=prod_curr_list,
                                                        debug= debug,
                                                        verbose=verbose)
                            continue

                        # get their respective sym reprs
                        next_sym_state: ADD = self.get_sym_state_from_tuple(next_ts_tuple)
                        next_dfa_sym: ADD = self.dfa_handle.dfa_predicate_add_sym_map_curr[next_dfa_tuple]
                        next_utls_sym : ADD = self.gou_handle.predicate_sym_map_utls[next_utls_tuple]

                        # compute the next prod state as 0-1 ADD
                        next_prod_sym_state: ADD = next_sym_state & next_dfa_sym & next_utls_sym & next_ba_sym

                        # if we havent added this edge already 
                        self.add_edge_to_action_tr(curr_state_tuple=curr_prod_tuple,
                                                   curr_state_sym=curr_prod_sym_state,
                                                   next_state_tuple=(*next_state_tuple, next_ba_int),
                                                   mod_act_dict=mod_act_dict,
                                                   nxt_state_sym=next_prod_sym_state,
                                                   robot_action_name=robot_act,
                                                   prod_curr_list=prod_curr_list,
                                                   valid_hact_list=no_human_move_edge,
                                                   debug=debug)

                        if verbose:
                            next_ts_exp_state = self.get_state_from_tuple(next_ts_tuple)
                            print(f"Adding Robot edge: "\
                                 f"({curr_ts_exp_states}, {curr_dfa_tuple}, {curr_utls_tuple}, {curr_br_tuple})" \
                                 f"-------{robot_act}------> ({next_ts_exp_state}, {next_dfa_tuple}, {next_utls_tuple}, {next_ba_int})")
                        
                        if not (next_dfa_sym & self.dfa_handle.sym_goal_state).isZero():
                            self.add_state_to_leaf_node(curr_utls_tuple=curr_utls_tuple,
                                                        next_ts_tuple=next_ts_tuple,
                                                        next_dfa_tuple=next_dfa_tuple,
                                                        next_utls_tuple=next_utls_tuple,
                                                        next_br_tuple=next_ba_int,
                                                        next_prod_sym_state=next_prod_sym_state,
                                                        only_leaf_nodes=print_leaf_nodes)
                            self.closed |= next_prod_sym_state

                            continue

                        # add them to their respective bucket. . .
                        self.open_list[next_utls_tuple].add((*next_state_tuple, next_ba_int))
                        self.closed |= next_prod_sym_state

            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == self.gou_handle.max_ts_action_cost:
                    print(f"Done Computing the Graph of Best Response! Accepting Leaf nodes {self.lcount}; Total states {self.scount}; Total edges {self.ecount}")
                    break
            
            layer += 1
                            