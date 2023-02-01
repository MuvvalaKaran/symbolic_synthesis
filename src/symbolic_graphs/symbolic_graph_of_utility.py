'''
 This scripts all the functions to construct the graph of utility. 
'''
import re
import sys
import time
import warnings

from bidict import bidict
from functools import reduce
from itertools import product
from collections import defaultdict
from typing import Union, List, Tuple, Dict

from cudd import Cudd, BDD, ADD

from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs


class SymbolicGraphOfUtility(DynWeightedPartitionedFrankaAbs):

    def __init__(self,
                 curr_vars: List[ADD],
                 lbl_vars: List[ADD],
                 state_utls_vars: List[ADD],
                 robot_action_vars: List[ADD],
                 human_action_vars: List[ADD],
                 task,
                 domain,
                 ts_state_map: dict,
                 ts_states: list,
                 manager: Cudd,
                 weight_dict: dict,
                 seg_actions: dict,
                 max_ts_action_cost: int,
                 ts_state_lbls: list,
                 dfa_state_vars: List[ADD],
                 sup_locs: List[str],
                 top_locs: List[str],
                 budget: int,
                 dfa_handle,
                 ts_handle,
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

        self.dfa_handle = dfa_handle
        self.ts_handle = ts_handle

        self.int_weight_dict = int_weight_dict

        # need these two during image computation 
        self.dfa_bdd_x_list: List[BDD] = [i.bddPattern() for i in dfa_state_vars]
        self.dfa_bdd_transition_fun_list: List[BDD] = [i.bddPattern() for i in self.dfa_handle.tr_state_adds]

        self.max_ts_action_cost: int = max_ts_action_cost
        self.sym_vars_ults: List[ADD] = state_utls_vars
        self.energy_budget: int = budget

        # index to determine where the state vars start 
        self.state_start_idx: int =  len(self.sym_vars_human) + len(self.sym_vars_robot)

        self.predicate_sym_map_utls: bidict = {}
        self.utls_cube = reduce(lambda x, y: x & y, self.sym_vars_ults)

        # overide the base class's adj dictionary
        self.prod_adj_map = defaultdict(lambda: defaultdict(lambda: {}))

        # the # of vars in the TS state has increased as we have additional varibale associted with state utility. 
        num_ts_state_vars: int = sum([len(listElem) for listElem in self.sym_vars_lbl]) + len(self.sym_vars_curr) + len(state_utls_vars)
        num_dfa_state_vars: int = len(dfa_state_vars)
        num_of_acts = len(list(self.tr_action_idx_map.keys()))
        self.sym_tr_actions = [[self.manager.addZero() for _ in range(num_ts_state_vars + num_dfa_state_vars)] for _ in range(num_of_acts)]

        # create state util look up dictionary
        self._initialize_add_for_state_utls()

        # count # of state in this graph
        self.scount = 0

        self.leaf_nodes: ADD = self.manager.plusInfinity()
        self.leaf_vals = set()
        self.lcount = 0
    

    def _initialize_add_for_state_utls(self):
        """
         This function initializes ADD that represents the state utility at each state. 
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
        
        # look up the corresonponding state int
        
        return curr_dfa_sym, self.dfa_handle.dfa_predicate_add_sym_map_curr.inv[curr_dfa_sym]
    

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
         Create the edge for the Graph of Utilitiy and store in a Compositional fashion. 
        """
        if valid_hact_list is not None:
            assert isinstance(valid_hact_list, ADD), "Error Constructing TR Edges. Fix This!!!"
            no_human_move = valid_hact_list
        
        if human_action_name != '':
            assert robot_action_name != '', "Error While constructing Human Edge, FIX THIS!!!"
        

        # get the modified robot action name
        mod_raction_name: str = mod_act_dict[robot_action_name]
        robot_move: ADD = self.ts_handle.predicate_sym_map_robot[mod_raction_name]

        curr_state_val: int = kwargs['curr_state_val']
        next_state_val: int = kwargs['next_state_val']
        sym_curr_state_val: ADD = self.predicate_sym_map_utls[curr_state_val]
        sym_next_state_val: ADD = self.predicate_sym_map_utls[next_state_val]

        _tr_idx: int = self.tr_action_idx_map.get(robot_action_name)

        if human_action_name != '':
            # get the modified human action name
            mod_haction_name: str = mod_act_dict[human_action_name]
            
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & sym_curr_state_val & robot_move & self.ts_handle.predicate_sym_map_human[mod_haction_name]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_str_state}[{curr_state_val}] --- {robot_action_name} & {human_action_name}---> {next_str_state}[{next_state_val}]")
                
                self.mono_tr_bdd |= curr_state_sym & sym_curr_state_val & robot_move & self.ts_handle.predicate_sym_map_human[mod_haction_name]
        else:
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & sym_curr_state_val & robot_move & no_human_move).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_str_state}[{curr_state_val}] ---{robot_action_name}---> {next_str_state}[{next_state_val}]")

                self.mono_tr_bdd |= curr_state_sym & sym_curr_state_val & robot_move & no_human_move

        nxt_prod_state_sym = nxt_state_sym & sym_next_state_val

        # generate all the cubes, with their corresponding string repr and leaf value (state value should be 1)
        add_cube: List[Tuple(list, int)] = list(nxt_prod_state_sym.generate_cubes())   
        assert len(add_cube) == 1, "Error computing cube string for next state's symbolic representation. FIX THIS!!!"
        assert add_cube[0][1] == 1, "Error computing next state cube. The integer value of the leaf node in the ADD is not 1. FIX THIS!!!"
        
        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(add_cube[0][0]):
            if var == 1 and self.manager.addVar(_idx) in kwargs['prod_curr_list']:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                # if human intervenes then the edge looks like (true) & (human move b# l# l#)
                if human_action_name != '':
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & self.ts_handle.predicate_sym_map_human[mod_haction_name] & sym_curr_state_val
                # if human does not intervene then the edge looks like (robot-action) & not(valid human moves)
                else:
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & no_human_move & sym_curr_state_val

            
            elif var == 2 and self.manager.addVar(_idx) in kwargs['prod_curr_list']:
                warnings.warn("Encountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)


        # create the adj map for rollout purposes
        if human_action_name != '':
            assert self.prod_adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get(mod_haction_name) is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.prod_adj_map[(*curr_state_tuple, curr_state_val)][mod_raction_name][mod_haction_name] = (*next_state_tuple, next_state_val)
        
        else:
            assert self.prod_adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get('r') is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.prod_adj_map[(*curr_state_tuple, curr_state_val)][mod_raction_name]['r'] = (*next_state_tuple, next_state_val)
        
        # update edge count 
        self.ecount += 1


    def construct_graph_of_utility(self, mod_act_dict: dict, boxes: List[str], verbose: bool = False, debug: bool = True):
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

        open_list = defaultdict(lambda: set())

        # keep track of states as we expand
        closed = self.manager.addZero()

        init_state_tuple = self.get_tuple_from_state(self.init)
        init_dfa_state = (self.dfa_handle.init[0])

        sym_trap_state_lbl: ADD = self.get_trap_state_lbl(boxes)

        layer = 0

        # creating monolithinc tr bdd to keep track all the transition we are creating to detect nondeterminism
        if debug:
            self.mono_tr_bdd = self.manager.addZero() 
        

        open_list[layer].add((init_state_tuple, init_dfa_state))

        prod_curr_list: List[ADD] = [*self.dfa_handle.sym_add_vars_curr]
        prod_curr_list.extend([lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list])
        prod_curr_list.extend([*self.sym_vars_curr, *self.sym_vars_ults])

        # used to break the loop
        empty_bucket_counter: int = 0
        
        while True:

            if len(open_list[layer]) > 0:
                if verbose:
                    print(f"********************Layer: {layer}**************************")
                
                # reset the empty bucket counter 
                empty_bucket_counter = 0

                # look over all the valid actions from each state from the current layer, 
                for curr_prod_tuple in open_list[layer]:
                    curr_state_tuple = curr_prod_tuple[0]
                    curr_dfa_tuple = curr_prod_tuple[1]
                    
                    # get the sym repr of the DFA state
                    curr_dfa_sym_state: ADD = self.dfa_handle.dfa_predicate_add_sym_map_curr[curr_dfa_tuple]
                    
                    # get sym repr of current and next state
                    curr_ts_sym_state: ADD = self.get_sym_state_from_tuple(curr_state_tuple)
                    curr_ts_exp_states = self.get_state_from_tuple(curr_state_tuple)

                    # create current sym prod state
                    curr_prod_sym_state: ADD = curr_ts_sym_state & curr_dfa_sym_state

                    # if the current prod state is an accepting state then add it to the leaf ADD along with the state value 
                    if not (curr_dfa_sym_state & self.dfa_handle.sym_goal_state).isZero():
                        print(f"Adding leaf node ({curr_ts_exp_states},{curr_dfa_tuple}) with value {layer}")
                        # before adding the leaf node, convert it to (state-val) - infinity ADD
                        full_prod_state_val: ADD = (curr_prod_sym_state & self.predicate_sym_map_utls[layer]).ite(self.manager.addConst(layer), self.manager.plusInfinity())
                        self.leaf_nodes = self.leaf_nodes.min(full_prod_state_val)
                        self.leaf_vals.add(layer)
                        # update counter
                        self.lcount += 1
                        continue
                        
                    # update the closed set
                    assert (closed & curr_prod_sym_state & self.predicate_sym_map_utls[layer]).isZero(), "Error unrolling the graph. Encountered a twice during unrolling. FIX THIS!!!"
                    closed |= curr_prod_sym_state & self.predicate_sym_map_utls[layer]

                    self.scount += 1

                    # loop over all valid action
                    for robot_act in self.ts_handle.org_adj_map[curr_state_tuple].keys():
                        # get the utility of the successor state
                        succ_utl: int = layer + self.int_weight_dict[robot_act]
                        
                        # edge corresponding to human intervention
                        no_human_move_edge: ADD = self.manager.addOne()

                        if succ_utl <= self.energy_budget:
                            valid_act_list = []
                            for human_act, next_exp_state in self.ts_handle.org_adj_map[curr_state_tuple][robot_act].items():
                                # first loop over all the human intervening move and then the non-intervening one

                                if human_act == 'r':
                                    continue
                                
                                # update list of valid human actions
                                valid_act_list.append(human_act)

                                next_sym_state: ADD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)
                                next_state_tuple = self.get_tuple_from_state(next_exp_state) 

                                # get the DFA state
                                next_dfa_sym, next_dfa_tuple = self.get_dfa_evolution(next_sym_state=next_sym_state, curr_dfa_sym=curr_dfa_sym_state)

                                # compute the next pros state
                                next_prod_sym_state: ADD = next_sym_state & next_dfa_sym

                                self.add_edge_to_action_tr(curr_state_tuple=curr_prod_tuple,
                                                           curr_state_sym=curr_prod_sym_state,
                                                           next_state_tuple=(next_state_tuple, next_dfa_tuple),
                                                           mod_act_dict=mod_act_dict,
                                                           nxt_state_sym=next_prod_sym_state,
                                                           human_action_name=human_act,
                                                           robot_action_name=robot_act,
                                                           curr_state_val=layer,
                                                           next_state_val=succ_utl,
                                                           prod_curr_list=prod_curr_list,
                                                           debug=debug)
                                
                                if verbose:
                                    print(f"Adding Human edge: ({curr_ts_exp_states}, {curr_dfa_tuple})[{layer}] -------{robot_act}{human_act}------> ({next_exp_state},{next_dfa_tuple})[{succ_utl}]")

                                
                                # add them to their respective bucket. . .
                                open_list[succ_utl].add((next_state_tuple, next_dfa_tuple))
                
                            # if there are any valid human edges from curr state
                            if len(valid_act_list) > 0:
                                valid_hact: List[ADD] = [self.ts_handle.predicate_sym_map_human[mod_act_dict[ha]] for ha in valid_act_list]
                                no_human_move_edge: ADD = ~(reduce(lambda x, y: x | y, valid_hact))    

                                assert not no_human_move_edge.isZero(), "Error computing a human no-intervene edge. FIX THIS!!!"
                            
                            next_exp_state: List[str] = self.ts_handle.org_adj_map[curr_state_tuple][robot_act]['r']
                            next_state_tuple = self.get_tuple_from_state(next_exp_state) 
                            next_sym_state: ADD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)

                            # get the DFA state
                            next_dfa_sym, next_dfa_tuple = self.get_dfa_evolution(next_sym_state=next_sym_state, curr_dfa_sym=curr_dfa_sym_state)

                            next_prod_sym_state: ADD = next_sym_state & next_dfa_sym

                            # create no-intervening edge
                            self.add_edge_to_action_tr(curr_state_tuple=curr_prod_tuple,
                                                       curr_state_sym=curr_prod_sym_state,
                                                       next_state_tuple=(next_state_tuple, next_dfa_tuple),
                                                       mod_act_dict=mod_act_dict,
                                                       nxt_state_sym=next_prod_sym_state,
                                                       robot_action_name=robot_act,
                                                       curr_state_val=layer,
                                                       next_state_val=succ_utl,
                                                       valid_hact_list=no_human_move_edge,
                                                       prod_curr_list=prod_curr_list,
                                                       debug=debug)
                            
                            if verbose:
                                print(f"Adding Robot edge: ({curr_ts_exp_states}, {curr_dfa_tuple})[{layer}] -------{robot_act}------> ({next_exp_state}, {next_dfa_tuple})[{succ_utl}]")
                            
                            # add them to their respective bucket. . .
                            open_list[succ_utl].add((next_state_tuple, next_dfa_tuple))

                        else:
                            # add an edge to the trap state
                            next_sym_state: ADD = self.get_sym_state_from_exp_states(exp_states=['(trap-state)'])
                            next_state_tuple: tuple = (self.pred_int_map['(trap-state)'])

                            next_prod_sym_state: ADD = next_sym_state & sym_trap_state_lbl & curr_dfa_sym_state

                            # add this (vT, lbl, DFA state) to the list leaf node
                            full_prod_state_val: ADD = (next_prod_sym_state & self.predicate_sym_map_utls[self.energy_budget]).ite(self.manager.addConst(self.energy_budget), self.manager.plusInfinity())
                            self.leaf_nodes = self.leaf_nodes.min(full_prod_state_val)

                            # update closed list for book keeping purposes
                            closed |= next_prod_sym_state & self.predicate_sym_map_utls[self.energy_budget]

                            self.add_edge_to_action_tr(curr_state_tuple=curr_prod_tuple,
                                                        next_state_tuple=(next_state_tuple, curr_dfa_tuple),
                                                        curr_state_sym=curr_prod_sym_state,
                                                        nxt_state_sym=next_prod_sym_state,
                                                        mod_act_dict=mod_act_dict,
                                                        robot_action_name=robot_act,
                                                        valid_hact_list=no_human_move_edge, 
                                                        curr_state_val=layer,
                                                        next_state_val=self.energy_budget,
                                                        prod_curr_list=prod_curr_list,
                                                        debug=debug)
                        
                            if verbose:
                                print(f"Adding Trap edge: ({curr_ts_exp_states}, {curr_dfa_tuple})[{layer}] -------{robot_act}------> (vT)")

                layer += 1
            
            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == self.max_ts_action_cost:
                    print(f"Done Computing the Graph of Utility! Accepting Leaf nodes {self.lcount}; Total states {self.scount}; Total edges {self.ecount}")
                    break


