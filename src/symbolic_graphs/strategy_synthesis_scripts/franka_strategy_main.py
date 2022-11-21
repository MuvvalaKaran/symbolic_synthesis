'''
This script implements Winning staregy and regret strategy synthesis code for Franka World 
'''
import sys
import time
import warnings

from collections import defaultdict
from typing import Union, List, Tuple

from cudd import Cudd, BDD, ADD

from src.explicit_graphs import CausalGraph

from src.algorithms.strategy_synthesis import ReachabilityGame

from src.symbolic_graphs import PartitionedDFA
from src.symbolic_graphs import PartitionedFrankaTransitionSystem, DynamicFrankaTransitionSystem

from src.symbolic_graphs.graph_search_scripts import FrankaWorld


class FrankaPartitionedWorld(FrankaWorld):
    """
     This base class construct a symbolic TR in a partitioned fashion. So, we do need two explicit set of boolean variables to
      contruct the TR. Addiitonally, we also add system's actions to the boolean formula. 
    """
    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 formulas: Union[List, str],
                 manager: Cudd,
                 algorithm: str,
                 weight_dict: dict = {},
                 ltlf_flag: bool = True,
                 dyn_var_ord: bool = False,
                 verbose: bool = False,
                 plot_ts: bool = False,
                 plot_obs: bool = False,
                 plot_dfa: bool = False,
                 plot: bool = False,
                 create_lbls: bool = True):
        super().__init__(domain_file, problem_file, formulas, manager, algorithm, weight_dict, ltlf_flag, dyn_var_ord, verbose, plot_ts, plot_obs, plot_dfa, plot, create_lbls)


    def build_abstraction(self, draw_causal_graph: bool = False, dynamic_env: bool = False):
        """
         A main function that construct a symbolic Franka World TS and its corresponsing DFA
        """
        print("*****************Creating Boolean variables for Partitioned Frankaworld!*****************")
        if self.algorithm == 'quant':
            # All vars (TS, DFA and Predicate) are of type ADDs
            raise NotImplementedError()
        
        elif self.algorithm == 'qual':
            if dynamic_env:
                sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars = self.build_bdd_abstraction_dynamic(draw_causal_graph=draw_causal_graph)
            else:
                sym_tr, ts_curr_vars, ts_action_vars, ts_lbl_vars = self.build_bdd_abstraction(draw_causal_graph=draw_causal_graph)

            dfa_tr, dfa_curr_state = self.build_bdd_symbolic_dfa(sym_tr_handle=sym_tr)
        
        else:
            warnings.warn("Please enter a valid graph search algorthim. Currently Available - Quanlitative")
        
        self.ts_handle: Union[DynamicFrankaTransitionSystem, PartitionedFrankaTransitionSystem] = sym_tr
        self.dfa_handle: PartitionedDFA = dfa_tr

        self.ts_x_list: List[BDD] = ts_curr_vars
        self.dfa_x_list: List[BDD] = dfa_curr_state
        self.ts_obs_list: List[BDD] = ts_lbl_vars

        if dynamic_env:
            self.ts_robot_vars: List[BDD] = ts_robot_vars
            self.ts_human_vars: List[BDD] = ts_human_vars
        
        else:
            warnings.warn("We haven't implemented a strategy synthesie for single player Partitioned Representation.")
            warnings.warn("Use the Monolithic Representation if you want graph search by setting the FRANKWORLD flag to True.")
            raise NotImplementedError()
        
        if self.dyn_var_ordering:
            self.set_variable_reordering()


    def create_symbolic_causal_graph(self, draw_causal_graph: bool = False, add_flag: bool = False, build_human_move: bool = False, print_facts: bool = False) -> Tuple:
        """
         Overrides the base method. We create boolean variables for all the actions possible.
          We also only create one set of variables (curr state vars) when constructing the symbolic Transition Relation   
        """
        _causal_graph_instance = CausalGraph(problem_file=self.problem_file,
                                             domain_file=self.domain_file,
                                             draw=draw_causal_graph)

        _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

        task_facts: List[str] = _causal_graph_instance.task.facts
        boxes: List[str] = _causal_graph_instance.task_objects

        # compute all valid preds of the robot conf and box conf.
        robot_preds, on_preds, box_preds = self.compute_valid_predicates(predicates=task_facts, boxes=boxes)

        # compute all the possible states
        ts_state_tuples = self.compute_valid_franka_state_tuples(robot_preds=robot_preds, on_preds=on_preds, verbose=True)
        
        if build_human_move:
            _seg_action = defaultdict(lambda: [])
            # segregate actions in robot actions (controllable vars - `o`) and humans moves (uncontrollable vars - `i`)
            for act in _causal_graph_instance.task.operators:
                if 'human' not in act.name:
                    _seg_action['robot'].append(act)
                else:
                    _seg_action['human'].append(act)
        
        if build_human_move:
            ts_human_act_vars = self._create_symbolic_lbl_vars(state_lbls=_seg_action['human'],
                                                               state_var_name='i',
                                                               add_flag=add_flag)

            ts_robot_act_vars = self._create_symbolic_lbl_vars(state_lbls=_seg_action['robot'],
                                                               state_var_name='o',
                                                               add_flag=add_flag)
            
            if print_facts:
                print(f"******************# of boolean Vars for Human actions: {len(ts_human_act_vars)}******************")
                print(f"******************# of boolean Vars for Robot actions: {len(ts_robot_act_vars)}******************")
        
        else:
            ts_action_vars = self._create_symbolic_lbl_vars(state_lbls=_causal_graph_instance.task.operators,
                                                            state_var_name='o',
                                                            add_flag=add_flag)
            
            if print_facts:
                print(f"******************# of boolean Vars for Robot actions: {len(ts_action_vars)}******************")

        
        # box_preds has predicates segregated as per boxes
        ts_lbl_vars = []
        for _id, b in enumerate(box_preds.keys()):
            if b == 'gripper':
                ts_lbl_vars.extend(self._create_symbolic_lbl_vars(state_lbls=box_preds[b],
                                                                  state_var_name=f'g_',
                                                                  add_flag=add_flag))
            else:
                ts_lbl_vars.extend(self._create_symbolic_lbl_vars(state_lbls=box_preds[b],
                                                                  state_var_name=f'b{_id}_',
                                                                  add_flag=add_flag))
        
        if print_facts:
            print(f"******************# of boolean Vars for TS lbls: {len(ts_lbl_vars)}******************")

        # The order for the boolean vara is first actions vars, then lbls, then state vars
        curr_vars = self._create_symbolic_lbl_vars(state_lbls=ts_state_tuples,
                                                   state_var_name='x',
                                                   add_flag=add_flag)
        
        if print_facts:
            print(f"******************# of boolean Vars for TS states: {len(curr_vars)}******************")
        
        if build_human_move:
            return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_vars, \
                 ts_state_tuples, ts_lbl_vars, ts_robot_act_vars, ts_human_act_vars, boxes, box_preds
        
        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_vars, ts_state_tuples, ts_lbl_vars, ts_action_vars, boxes, box_preds
    

    def build_bdd_abstraction(self, draw_causal_graph: bool = False) -> Tuple[PartitionedFrankaTransitionSystem, List[BDD], List[BDD], List[BDD]]:
        """
         Main Function to Build Transition System that only represent valid edges without any weights
        """
        task, domain, ts_curr_vars, ts_state_tuples, ts_lbl_vars, ts_action_vars, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                                                                           print_facts=True)

        sym_tr = PartitionedFrankaTransitionSystem(curr_vars=ts_curr_vars,
                                                   lbl_vars=ts_lbl_vars,
                                                   action_vars=ts_action_vars,
                                                   task=task,
                                                   domain=domain,
                                                   ts_states=ts_state_tuples,
                                                   ts_state_map=self.pred_int_map,
                                                   manager=self.manager)
        start: float = time.time()
        sym_tr.create_transition_system_franka(boxes=boxes,
                                               state_lbls=possible_lbls,
                                               add_exist_constr=True,
                                               verbose=True,
                                               plot=self.plot_ts)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)

        return sym_tr, ts_curr_vars, ts_action_vars, ts_lbl_vars
    

    def build_bdd_abstraction_dynamic(self, draw_causal_graph: bool = False) -> Tuple[PartitionedFrankaTransitionSystem, List[BDD], List[BDD], List[BDD], List[BDD]]:
        """
         Main Function to Build Two-player Transition System without edge weights
        """
        task, domain, ts_curr_vars, ts_state_tuples, \
             ts_lbl_vars, ts_robot_vars, ts_human_vars, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                                                 build_human_move=True,
                                                                                                                 print_facts=True)
        
        sym_tr = DynamicFrankaTransitionSystem(curr_vars=ts_curr_vars,
                                               lbl_vars=ts_lbl_vars,
                                               robot_action_vars=ts_robot_vars,
                                               human_action_vars=ts_human_vars,
                                               task=task,
                                               domain=domain,
                                               ts_states=ts_state_tuples,
                                               ts_state_map=self.pred_int_map,
                                               manager=self.manager)
        
        start: float = time.time()
        sym_tr.create_transition_system_franka(boxes=boxes,
                                               state_lbls=possible_lbls,
                                               add_exist_constr=True,
                                               verbose=self.verbose,
                                               plot=self.plot_ts,
                                               print_tr=False)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)
        
        return sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars
    

    def build_bdd_symbolic_dfa(self, sym_tr_handle: Union[DynamicFrankaTransitionSystem, PartitionedFrankaTransitionSystem]) \
         -> Tuple[List[PartitionedDFA], List[BDD], List[BDD]]:
        """
         This function constructs the DFA in a partitioned fashion. We do not need two sets of variables to construct the transition relations.
        """
        if len(self.formulas) > 1:
            warnings.warn("Trying to construt Partitioned DFA representation for multiple Formulas. This functionality only works for Monolithic represrntation.")
            sys.exit(-1)

        dfa_curr_state, _dfa = self.create_partitioned_symbolic_dfa_graph(formula=self.formulas[0])

        start = time.time()
        # create TR corresponding to each DFA - dfa name is only used dumping graph 
        dfa_tr = PartitionedDFA(curr_states=dfa_curr_state,
                                predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                sym_tr=sym_tr_handle,
                                manager=self.manager,
                                dfa=_dfa,
                                ltlf_flag=self.ltlf_flag,
                                dfa_name='dfa_0')
        if self.ltlf_flag:
            dfa_tr.create_symbolic_ltlf_transition_system(verbose=self.verbose, plot=self.plot_dfa)
        else:
            raise NotImplementedError()
        stop = time.time()
        print("Time for Constructing the LTLF DFA: ", stop - start)

        return dfa_tr, dfa_curr_state
    

    def set_variable_reordering(self, make_tree_node: bool = False, **kwargs):
        """
         Overides the parent method and removes the TREE node computation
          as we do not have two explicit set of variables for curr state and next state vars in our Partitioned TR representation
        """
        self.manager.autodynEnable()

        if self.verbose:
            self.manager.enableOrderingMonitoring()
        else:
            self.manager.enableReorderingReporting()
    

    def solve(self, verbose: bool = False) -> BDD:
        """
         A function that call the winning strategy synthesis code and compute the set of winnign states and winning strategy for robot. 
        """
        start = time.time()
        reachability_handle =  ReachabilityGame(ts_handle=self.ts_handle,
                                                 dfa_handle=self.dfa_handle,
                                                 ts_curr_vars=self.ts_x_list,
                                                 dfa_curr_vars=self.dfa_x_list,
                                                 ts_obs_vars=self.ts_obs_list,
                                                 sys_act_vars=self.ts_robot_vars,
                                                 env_act_vars=self.ts_human_vars,
                                                 cudd_manager=self.manager)

        win_str: BDD = reachability_handle.solve(verbose=verbose)
        stop = time.time()
        print("Time for solving the game: ", stop - start)

        # rollout for sanity checking
        reachability_handle.roll_out_strategy(transducer=win_str, verbose=True)

        return win_str