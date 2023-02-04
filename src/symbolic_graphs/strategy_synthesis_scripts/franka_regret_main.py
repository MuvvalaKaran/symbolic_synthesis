import re
import sys
import time
import math
import warnings


from bidict import bidict
from collections import defaultdict
from typing import Union, List, Tuple, Dict

from cudd import Cudd, BDD, ADD

from src.explicit_graphs import CausalGraph, Ltlf2MonaDFA

from src.algorithms.strategy_synthesis import AdversarialGame, GraphOfUtlCooperativeGame

from src.symbolic_graphs import PartitionedDFA, ADDPartitionedDFA
from src.symbolic_graphs import PartitionedFrankaTransitionSystem, DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem
from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs
from src.symbolic_graphs import SymbolicGraphOfUtility

from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld


class FrankaRegretSynthesis(FrankaPartitionedWorld):
    """
      Main script that constructs the Main Graph in a partitioned fashion, then constructs the graph of utility (G^{u}) and
       finally the Graph of best response (G^{br}). 
    """

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 formulas: Union[List, str],
                 manager: Cudd,
                 algorithm: str,
                 sup_locs: List[str],
                 top_locs: List[str],
                 weight_dict: dict = {},
                 ltlf_flag: bool = True,
                 dyn_var_ord: bool = False,
                 verbose: bool = False,
                 plot_ts: bool = False,
                 plot_obs: bool = False,
                 plot_dfa: bool = False,
                 plot: bool = False,
                 create_lbls: bool = True):
        super().__init__(domain_file=domain_file,
                         problem_file=problem_file,
                         formulas=formulas,
                         manager=manager,
                         algorithm=algorithm,
                         sup_locs=sup_locs,
                         top_locs=top_locs,
                         weight_dict=weight_dict,
                         ltlf_flag=ltlf_flag,
                         dyn_var_ord=dyn_var_ord,
                         verbose=verbose,
                         plot_ts=plot_ts,
                         plot_obs=plot_obs,
                         plot_dfa=plot_dfa,
                         plot=plot,
                         create_lbls=create_lbls)
        
        # graph of utility handle
        self.graph_of_utls_handle: SymbolicGraphOfUtility = None

        # Map to store org act name to mod act name
        # create during the first abstraction construction call
        self.mod_act_dict = None

        # maps each action to int weight
        self.int_weight_dict = None
        self.task_boxes: List[str] = [] 

        self.min_energy_budget: Union[int, float] = math.inf
        self.reg_energy_budget: Union[int, float] = math.inf

        # keep track of utility and best alternative response
        self.prod_utls_vars: List[ADD] = None
        self.prod_ba_vars: List[ADD] = None
    

    def build_abstraction(self, draw_causal_graph: bool = False, dynamic_env: bool = False, bnd_dynamic_env: bool = False, max_human_int: int = 0):
        """
         Main function that constructs the Original graph G and the DFA associated with LTLf formula. 
        """
        sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars = self.build_add_abstraction_dynamic(draw_causal_graph=draw_causal_graph)

        dfa_tr = self.build_add_symbolic_dfa(sym_tr_handle=sym_tr)

        self.ts_handle: DynWeightedPartitionedFrankaAbs = sym_tr
        self.dfa_handle: ADDPartitionedDFA = dfa_tr

        self.ts_x_list: List[ADD] = ts_curr_vars
        self.ts_obs_list: List[ADD] = ts_lbl_vars

        self.ts_robot_vars: List[ADD] = ts_robot_vars
        self.ts_human_vars: List[ADD] = ts_human_vars
    

    def create_symbolic_causal_graph(self, draw_causal_graph: bool = False, add_flag: bool = False, build_human_move: bool = True, print_facts: bool = False) -> Tuple:
        """
         We overide PartitionedFrankaWorld's method to add an additional state - Terminal state 
        """

        _causal_graph_instance = CausalGraph(problem_file=self.problem_file,
                                             domain_file=self.domain_file,
                                             draw=draw_causal_graph)

        _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

        task_facts: List[str] = _causal_graph_instance.task.facts
        boxes: List[str] = _causal_graph_instance.task_objects

        # compute all valid preds of the robot conf and box conf.
        robot_preds, box_preds = self.compute_valid_predicates(predicates=task_facts, boxes=boxes)

        # add a terminal-state to the set of valid robot preds
        robot_preds.append('(trap-state)')

        # box_preds.append('(empty)')

        # update the predicate dictionary accordingly
        num_of_preds: int = len(self.pred_int_map.keys())
        self.pred_int_map['(trap-state)'] = num_of_preds
        num_of_preds += 1

        # throw warning if there is exactly one human (hbox) location
        if len(_causal_graph_instance.task_intervening_locations) <= 1:
            warnings.warn("If you do not want human interventions then have only 1 hbox loc. \
        Ensure you have atleast two hbox locs for human intervention (edges) to exists.")
            if len(_causal_graph_instance.task_intervening_locations) == 0:
                sys.exit(-1)
        
        # segregate actions in robot actions (controllable vars - `o`) and humans moves (uncontrollable vars - `i`)
        _seg_action = self.get_seg_human_robot_action(_causal_graph_instance)
        
        ts_human_act_vars = self._create_symbolic_lbl_vars(state_lbls=_seg_action['human'],
                                                            state_var_name='i',
                                                            add_flag=add_flag)

        ts_robot_act_vars = self._create_symbolic_lbl_vars(state_lbls=_seg_action['robot'],
                                                            state_var_name='o',
                                                            add_flag=add_flag)
        
        if print_facts:
            print(f"******************# of boolean Vars for Human actions: {len(ts_human_act_vars)}******************")
            print(f"******************# of boolean Vars for Robot actions: {len(ts_robot_act_vars)}******************")

        # build DFA State vars
        dfa_state_vars, dfa_tr = self.create_partitioned_symbolic_dfa_graph(formula=self.formulas[0],
                                                                            add_flag=add_flag)
        self.dfa_ltlf_handle: Ltlf2MonaDFA = dfa_tr
        self.dfa_x_list = dfa_state_vars

        if print_facts:
            print(f"******************# of boolean Vars for DFA state: {len(dfa_state_vars)}******************")

        
        # box_preds has predicates segregated as per boxes
        ts_lbl_vars = []
        for _id, b in enumerate(box_preds.keys()):
            # We also add an `empty` label that only correpsonds to the trap state.
            # As we create dedicated bVars for each box, we have to create empty for each box
            box_preds[b] = box_preds[b] + [f'(on b{_id} empty)']
            # update the predicate int map
            self.pred_int_map[f'(on b{_id} empty)'] = num_of_preds
            num_of_preds += 1
            ts_lbl_vars.append(self._create_symbolic_lbl_vars(state_lbls=box_preds[b],
                                                              state_var_name=f'b{_id}_',
                                                              add_flag=add_flag))                                                   
        
        if print_facts:
            count = sum([len(listElem) for listElem in ts_lbl_vars])
            print(f"******************# of boolean Vars for TS lbls: {count}******************")

        # The order for the boolean vars is first actions vars, then lbls, then state vars
        curr_vars = self._create_symbolic_lbl_vars(state_lbls=robot_preds,
                                                   state_var_name='x',
                                                   add_flag=add_flag)
        
        if print_facts:
            print(f"******************# of boolean Vars for Robot Conf: {len(curr_vars)}******************")
        
        
        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, _causal_graph_instance, curr_vars, \
                robot_preds, ts_lbl_vars, ts_robot_act_vars, ts_human_act_vars, _seg_action, boxes, box_preds

    
    def _create_weight_dict(self, mod_action: dict, **kwargs) -> Dict[str, int]:
        """
         Override the base method make action ti and within human region twice as expensive as the robot region. 
        """
        causal_instance: CausalGraph = kwargs['causal_graph']
        task = kwargs['task']

        _loc_pattern = "[l|L][\d]+"

        new_weight_dict = {}
        for op in task.operators:
            # extract the action name
            if 'transit' in op.name:
                locs: List[str] = re.findall(_loc_pattern, op.name)

                if 'else' in op.name:
                    # you can transit from else to else. In such cases locs is empty.
                    if len(locs) == 0:
                        weight: int = self.weight_dict['transit']
                    elif locs[0] in causal_instance.task_intervening_locations:
                        weight: int = self.weight_dict['transit']
                    else:
                        weight: int = 2 * self.weight_dict['transit']
                else:
                    if locs[1] in causal_instance.task_intervening_locations:
                        weight: int = self.weight_dict['transit']     
                    else:
                        weight: int = 2 * self.weight_dict['transit']

                
            elif 'transfer' in op.name:
                locs: List[str] = re.findall(_loc_pattern, op.name)
                if 'else' in op.name:
                    if locs[0] in causal_instance.task_intervening_locations:
                        weight: int = self.weight_dict['transfer']
                    else:
                        weight: int = 2 * self.weight_dict['transfer']
                else:
                    if locs[1] in causal_instance.task_intervening_locations:
                        weight: int = self.weight_dict['transfer']
                    else:
                        weight: int = 2 * self.weight_dict['transfer']
            
            elif 'grasp' in op.name:
                locs: List[str] = re.findall(_loc_pattern, op.name)
                if 'else' in op.name:
                    weight: int = self.weight_dict['grasp']
                elif locs[0] in causal_instance.task_intervening_locations:
                    weight: int = self.weight_dict['grasp']
                else:
                    weight: int = 2 * self.weight_dict['grasp']
            
            elif 'release' in op.name:
                locs: List[str] = re.findall(_loc_pattern, op.name)
                if locs[0] in causal_instance.task_intervening_locations:
                    weight: int = self.weight_dict['release']
                else:
                    weight: int = 2 * self.weight_dict['release']
                
            else:
                weight: int = self.weight_dict['human']

            new_weight_dict[op.name] = weight

        return new_weight_dict
    

    def build_add_abstraction_dynamic(self, draw_causal_graph: bool = False, print_facts: bool = True) \
         -> Tuple[DynWeightedPartitionedFrankaAbs, List[ADD], List[ADD], List[ADD], List[ADD]]:
        """
         Main Function to Build Two-player Transition System with edge weights.
        """
        task, domain, causal_graph, ts_curr_vars, ts_state_tuples, \
             ts_lbl_vars, ts_robot_vars, ts_human_vars, modified_actions, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                                                                   build_human_move=True,
                                                                                                                                   print_facts=print_facts,
                                                                                                                                   add_flag=True)
        
        self.task_boxes = boxes
        self.mod_act_dict: dict = self.get_act_to_mod_act_dict(task=task)
        
        # get the actual parameterized actions and add their corresponding weights
        self.int_weight_dict = self._create_weight_dict(mod_action=modified_actions['robot'], task=task, causal_graph=causal_graph)
        
        # sort them according to their weights and then convert them in to addConst; reverse will sort the weights in descending order
        weight_dict = {k: v for k, v in sorted(self.int_weight_dict.items(), key=lambda item: item[1], reverse=True)}
        for action, w in weight_dict.items():
            weight_dict[action] = self.manager.addConst(int(w))
        
        sym_tr = DynWeightedPartitionedFrankaAbs(curr_vars=ts_curr_vars,
                                                 lbl_vars=ts_lbl_vars,
                                                 robot_action_vars=ts_robot_vars,
                                                 human_action_vars=ts_human_vars,
                                                 weight_dict=weight_dict,
                                                 task=task,
                                                 domain=domain,
                                                 seg_actions=modified_actions,
                                                 ts_states=ts_state_tuples,
                                                 ts_state_lbls=possible_lbls,
                                                 dfa_state_vars=self.dfa_x_list,
                                                 ts_state_map=self.pred_int_map,
                                                 sup_locs=self.sup_locs,
                                                 top_locs=self.top_locs,
                                                 manager=self.manager)
        
        start: float = time.time()
        sym_tr.create_transition_system_franka(boxes=boxes,
                                               add_exist_constr=True,
                                               verbose=self.verbose,
                                               plot=self.plot_ts,
                                               print_tr=False,
                                               debug=True,
                                               mod_act_dict=self.mod_act_dict)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)
        
        if print_facts:
            print(f"******************# of Edges in Franka Abstraction: {sym_tr.ecount}******************")

        return sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars
    

    def build_add_graph_of_utility(self, verbose: bool = False):
        """
         Main method that first plays the min-max game over the original graph. 
        
            This is the minimum budget we need to provide to ensure regret minimizing strategies exists. This graph (G')
            is of the form Target weighted arena (TWA).  The utility information is added to the nodes of G to construct
            the nodes of G'. The utility information added to the nodes is uniquely determined by the path used to reach
            the current  position.
        """
        # compute the minmax value of the game
        min_max_handle = AdversarialGame(ts_handle=self.ts_handle,
                                         dfa_handle=self.dfa_handle,
                                         ts_curr_vars=self.ts_x_list,
                                         dfa_curr_vars=self.dfa_x_list,
                                         ts_obs_vars=self.ts_obs_list,
                                         sys_act_vars=self.ts_robot_vars,
                                         env_act_vars=self.ts_human_vars,
                                         cudd_manager=self.manager)
        
        win_str: ADD = min_max_handle.solve(verbose=verbose)

        if win_str:
            if True:
                min_max_handle.roll_out_strategy(strategy=win_str, verbose=False)

        # min max value
        self.min_energy_budget = min_max_handle.init_state_value
        assert min_max_handle != math.inf, "No winning strategy exists. Before running regret game, make sure there existd a winning strategy."

        # regret budget
        self.reg_energy_budget = math.ceil(self.min_energy_budget * 2.0)

        # construct additional boolean variables used during the construction of the new graph
        self.prod_utls_vars = self._create_symbolic_lbl_vars(state_lbls=list(range(self.reg_energy_budget + 1)),
                                                                 state_var_name='k',
                                                                 add_flag=True)

        print(f"# of States in the Original graph: {len(self.ts_handle.adj_map.keys())}")

        # get the max action cost
        max_action_cost: int = min_max_handle._get_max_tr_action_cost()

        # construct the graph of utilty
        graph_of_utls_handle = SymbolicGraphOfUtility(curr_vars=self.ts_x_list,
                                                      lbl_vars=self.ts_obs_list,
                                                      state_utls_vars=self.prod_utls_vars,
                                                      robot_action_vars=self.ts_robot_vars,
                                                      human_action_vars=self.ts_human_vars,
                                                      task=self.ts_handle.task,
                                                      domain=self.ts_handle.domain,
                                                      ts_state_map=self.ts_handle.pred_int_map,
                                                      ts_states=self.ts_handle.ts_states,
                                                      manager=self.manager,
                                                      weight_dict=self.ts_handle.weight_dict,
                                                      seg_actions=self.ts_handle.actions,
                                                      max_ts_action_cost=max_action_cost,
                                                      ts_state_lbls=self.ts_handle.state_lbls,
                                                      dfa_state_vars=self.dfa_x_list,
                                                      sup_locs=self.sup_locs,
                                                      top_locs=self.top_locs,
                                                      dfa_handle=self.dfa_handle,
                                                      ts_handle=self.ts_handle,
                                                      int_weight_dict=self.int_weight_dict,
                                                      budget=self.reg_energy_budget)
        
        start: float = time.time()
        graph_of_utls_handle.construct_graph_of_utility(mod_act_dict=self.mod_act_dict,
                                                        boxes=self.task_boxes,
                                                        verbose=False,
                                                        debug=True)
        stop: float = time.time()
        print("Time took for constructing the Graph of Utility: ", stop - start)

        self.graph_of_utls_handle = graph_of_utls_handle

        print(f"************************** Energy Budget: {self.reg_energy_budget} **************************")
    

    def solve(self, verbose: bool = False):
        """
         OVerides base method to first construct the required graph and then run ValueIteration
        """

        # constuct graph of utility
        self.build_add_graph_of_utility(verbose=verbose)

        # compute the min-min value from each state
        gou_min_min_handle = GraphOfUtlCooperativeGame(prod_handle=self.graph_of_utls_handle,
                                                        ts_handle=self.ts_handle,
                                                        dfa_handle=self.dfa_handle,
                                                        ts_curr_vars=self.ts_x_list,
                                                        dfa_curr_vars=self.dfa_x_list,
                                                        sys_act_vars=self.ts_robot_vars,
                                                        env_act_vars=self.ts_human_vars,
                                                        ts_obs_vars=self.ts_obs_list,
                                                        ts_utls_vars=self.prod_utls_vars,
                                                        cudd_manager=self.manager)
        
        # compute the cooperative value from each prod state in the graph of utility
        cvals: ADD = gou_min_min_handle.solve(verbose=False)

        assert gou_min_min_handle.init_state_value == self.min_energy_budget, "Error computing CVal on Graph of Utility. Mismatch in Init stat value. Fix This!!!"

        start: float = time.time()
        # compute the best alternative from each edge for cumulative payoff
        self.graph_of_utls_handle.get_best_alternatives(cooperative_vals=cvals, mod_act_dict=self.mod_act_dict, verbose=False)
        stop: float = time.time()
        print("Time took for computing the set of best alternatives: ", stop - start)

        # construct addiotnal boolean vars for set of best alternative values
        ts_ba_vars: List[ADD] = self._create_symbolic_lbl_vars(state_lbls=self.graph_of_utls_handle.ba_set,
                                                               state_var_name='r',
                                                               add_flag=True)


        # we do need more boolean variables than the # of utility vars as ba_set is subset of utls vars
        assert len(ts_ba_vars) <= len(self.prod_utls_vars), "We should NOT create more boolean vars for Best alternative than Utility Vars. Fix this!!!"

        # compute the best alternative from each edge for cumulative payoff
        print("Done!!")