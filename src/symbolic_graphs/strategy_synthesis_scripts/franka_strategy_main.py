import re
import sys
import time
import warnings

from bidict import bidict
from collections import defaultdict
from typing import Union, List, Tuple, Dict

from cudd import Cudd, BDD, ADD

from src.explicit_graphs import CausalGraph, Ltlf2MonaDFA

from src.algorithms.strategy_synthesis import ReachabilityGame, BndReachabilityGame
from src.algorithms.strategy_synthesis import AdversarialGame, CooperativeGame
from src.algorithms.strategy_synthesis import QuantitativeBestEffortReachSyn

from src.symbolic_graphs import PartitionedDFA, ADDPartitionedDFA
from src.symbolic_graphs import PartitionedFrankaTransitionSystem, DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem
from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs
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
        # maps each action to int weight
        self.int_weight_dict: Dict[str, int]  = None

    
    def build_abstraction(self, draw_causal_graph: bool = False, dynamic_env: bool = False, bnd_dynamic_env: bool = False, max_human_int: int = 0):
        """
         A main function that construct a symbolic Franka World TS and its corresponsing DFA
        """
        print("*****************Creating Boolean variables for Partitioned Frankaworld!*****************")
        if 'quant' in self.algorithm:
            # All vars (TS, DFA and Predicate) are of type ADDs
            # unbounded human interventions
            if dynamic_env:
                sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars = self.build_add_abstraction_dynamic(draw_causal_graph=draw_causal_graph)
            
            # bounded human intervention as explicit state counter
            elif bnd_dynamic_env:
                raise NotImplementedError()
            
            dfa_tr = self.build_add_symbolic_dfa(sym_tr_handle=sym_tr)
        
        elif self.algorithm == 'qual':
            # unbounded human interventions
            if dynamic_env:
                sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars = self.build_bdd_abstraction_dynamic(draw_causal_graph=draw_causal_graph)
            
            # bounded human intervention as explicit state counter
            elif bnd_dynamic_env:
                sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars = self.build_bdd_bnd_abstraction_dynamic(draw_causal_graph=draw_causal_graph,
                                                                                                                         max_human_int=max_human_int)
            else:
                sym_tr, ts_curr_vars, ts_action_vars, ts_lbl_vars = self.build_bdd_abstraction(draw_causal_graph=draw_causal_graph)
            
            dfa_tr = self.build_bdd_symbolic_dfa(sym_tr_handle=sym_tr)
        
        else:
            warnings.warn("Please enter a valid graph search algorthim. Currently Available -'qual' or 'quant-adv' or 'quant-coop'")
            sys.exit(-1)
        
        self.ts_handle: Union[DynamicFrankaTransitionSystem, PartitionedFrankaTransitionSystem] = sym_tr
        self.dfa_handle: Union[PartitionedDFA, ADDPartitionedDFA] = dfa_tr

        self.ts_x_list: Union[List[BDD], List[ADD]] = ts_curr_vars
        self.ts_obs_list: Union[List[BDD], List[ADD]] = ts_lbl_vars

        if dynamic_env or bnd_dynamic_env:
            self.ts_robot_vars: Union[List[BDD], List[ADD]] = ts_robot_vars
            self.ts_human_vars: Union[List[BDD], List[ADD]] = ts_human_vars
        
        else:
            warnings.warn("We haven't implemented a strategy synthesis for single player Partitioned Representation.")
            warnings.warn("Use the Monolithic Representation if you want graph search by setting the FRANKAWORLD flag to True.")
            raise NotImplementedError()
        
        if self.dyn_var_ordering:
            self.set_variable_reordering()
    

    def get_seg_human_robot_action(self, causal_graph_instance: CausalGraph) -> dict:
        """
         A function that loops over all the actions and store them in a dict. 
        """
        _seg_action = defaultdict(lambda: [])

        for b in causal_graph_instance.task_objects:
            for l in causal_graph_instance.task_intervening_locations:
                _seg_action['human'].append(f'human-move {b} {l}')

        # for release and grasp, we only create one action
        # for transit we only create transit b# and for transfer l2#
        _seg_action['robot'].append('release') 
        _seg_action['robot'].append('grasp')
        for b in causal_graph_instance.task_objects:
            _seg_action['robot'].append(f'transit {b}')

        for l in causal_graph_instance.task_locations:
            _seg_action['robot'].append(f'transfer {l}')       
        
        return _seg_action
    

    def _create_weight_dict(self, mod_action, **kwargs) -> Dict[str, int]:
        """
         A function that loop over all the paramterized manipulator actions and
          assigns their corresponding weights from the weight dictionary specified as input.
        """
        task = kwargs["task"]
        new_weight_dict = {}
        for op in task.operators:
            # extract the action name
            if 'transit' in op.name:
                weight: int = self.weight_dict['transit']
            elif 'transfer' in op.name:
                weight: int = self.weight_dict['transfer']
            elif 'grasp' in op.name:
                weight: int = self.weight_dict['grasp']
            elif 'release' in op.name:
                weight: int = self.weight_dict['release']
            else:
                weight: int = self.weight_dict['human']
            
            new_weight_dict[op.name] = weight

        return new_weight_dict


    def get_act_to_mod_act_dict(self, task) -> Dict[str, str]:
        """
         A function that creates a mapping from the actual action name to modified action name
        """
        _org_to_mod_act = {}

        _loc_pattern = "[l|L][\d]+"
        _box_pattern = "[b|B][\d]+"
        for op in task.operators:
            if 'release' in op.name:
                _org_to_mod_act[op.name] = 'release'
            elif 'grasp' in op.name:
                _org_to_mod_act[op.name] = 'grasp'
            
            elif 'transit' in op.name:
                _box_state: str = re.search(_box_pattern, op.name).group()
                _org_to_mod_act[op.name] = f'transit {_box_state}'
                
            elif 'transfer' in op.name:
                locs: List[str] = re.findall(_loc_pattern, op.name)
                if 'else' in op.name:
                    _org_to_mod_act[op.name] = f'transfer {locs[0]}'
                else:
                    _org_to_mod_act[op.name] = f'transfer {locs[1]}'
            
            elif 'human' in op.name:
                locs: List[str] = re.findall(_loc_pattern, op.name)
                _box_state: str = re.search(_box_pattern, op.name).group()

                _org_to_mod_act[op.name] = f'human-move {_box_state} {locs[1]}'
            
            else:
                warnings.warn("Could not look up the corresponding modified robot action name")
                sys.exit(-1)

        return _org_to_mod_act
    

    def compute_valid_predicates(self, predicates: List[str], boxes: List[str]) -> Tuple[List, Dict]:
        """
        A helper function that segretaes the predicates as required by the symbolic transition relation. We separate them based on

         1) all gripper predicates - we do not need to create prime version for these
         2) all on predicates - we do need to create prime version for these
         3) all holding predicates 
         4) rest of the predicates - holding, ready, to-obj, and to-loc predicates. We do create prime version for these. 
        """

        predicate_dict = {
            'ready_all': [],
            'holding_all': [],
            'to_obj_all': [],
            'on': [],
        }

        # dictionary where we segreate on predicates based on boxes - all b0, b1 ,... into seperate list 
        boxes_dict = {box: [] for box in boxes} 

        for pred in predicates:
            if 'on' in pred:
                predicate_dict['on'].append(pred)
                for b in boxes:
                    if b in pred:
                        boxes_dict[b].append(pred)
                        break
            else:
                if 'holding' in pred:
                    predicate_dict['holding_all'].append(pred)
                if 'ready' in pred:
                    predicate_dict['ready_all'].append(pred)
                elif 'to-obj' in pred:
                    predicate_dict['to_obj_all'].append(pred)
        
        # create predicate int map
        _ind_pred_list = predicate_dict['ready_all'] + predicate_dict['holding_all'] + predicate_dict['to_obj_all']
        _pred_map = {pred: num for num, pred in enumerate(_ind_pred_list)}
        _pred_map = bidict(_pred_map)

        # create on predicate map
        len_robot_conf = len(_ind_pred_list)
        _pred_map.update({pred: len_robot_conf + num for num, pred in enumerate(predicate_dict['on'])})
        
        self.pred_int_map = _pred_map

        return _ind_pred_list, boxes_dict


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
        robot_preds, box_preds = self.compute_valid_predicates(predicates=task_facts, boxes=boxes)

        # throw warning if there is exactly one human (hbox) location
        if len(_causal_graph_instance.task_intervening_locations) <= 1:
            warnings.warn("If you do not want human interventions then have only 1 hbox loc. \
        Ensure you have atleast two hbox locs for human intervention (edges) to exists.")
            if len(_causal_graph_instance.task_intervening_locations) == 0:
                sys.exit(-1)
        
        # segregate actions in robot actions (controllable vars - `o`) and humans moves (uncontrollable vars - `i`)
        if build_human_move:
            _seg_action = self.get_seg_human_robot_action(_causal_graph_instance)
        
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
        
        if build_human_move:
            return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_vars, \
                 robot_preds, ts_lbl_vars, ts_robot_act_vars, ts_human_act_vars, _seg_action, boxes, box_preds
        
        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_vars, robot_preds, ts_lbl_vars, ts_action_vars, boxes, box_preds
    

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
    

    def build_bdd_abstraction_dynamic(self, draw_causal_graph: bool = False, print_facts: bool = True) \
         -> Tuple[DynamicFrankaTransitionSystem, List[BDD], List[BDD], List[BDD], List[BDD]]:
        """
         Main Function to Build Two-player Transition System without edge weights.
        """
        task, domain, ts_curr_vars, ts_state_tuples, \
             ts_lbl_vars, ts_robot_vars, ts_human_vars, modified_actions, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                                                                   build_human_move=True,
                                                                                                                                   print_facts=print_facts)
        
        sym_tr = DynamicFrankaTransitionSystem(curr_vars=ts_curr_vars,
                                               lbl_vars=ts_lbl_vars,
                                               robot_action_vars=ts_robot_vars,
                                               human_action_vars=ts_human_vars,
                                               task=task,
                                               domain=domain,
                                               ts_states=ts_state_tuples,
                                               ts_state_lbls=possible_lbls,
                                               dfa_state_vars=self.dfa_x_list,
                                               ts_state_map=self.pred_int_map,
                                               manager=self.manager,
                                               modified_actions=modified_actions,
                                               sup_locs=self.sup_locs,
                                               top_locs=self.top_locs)
        
        start: float = time.time()
        sym_tr.create_transition_system_franka(boxes=boxes,
                                               add_exist_constr=True,
                                               verbose=self.verbose,
                                               plot=self.plot_ts,
                                               print_tr=False,
                                               debug=True)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)
        
        if print_facts:
            print(f"******************# of Edges in Franka Abstraction: {sym_tr.ecount}******************")
            print(f"******************# of States in the Original graph: {len(sym_tr.adj_map.keys())}******************")

        return sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars
    

    def build_add_abstraction_dynamic(self, draw_causal_graph: bool = False, print_facts: bool = True) \
         -> Tuple[DynWeightedPartitionedFrankaAbs, List[ADD], List[ADD], List[ADD], List[ADD]]:
        """
         Main Function to Build Two-player Transition System with edge weights.
        """
        task, domain, ts_curr_vars, ts_state_tuples, \
             ts_lbl_vars, ts_robot_vars, ts_human_vars, modified_actions, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                                                                   build_human_move=True,
                                                                                                                                   print_facts=print_facts,
                                                                                                                                   add_flag=True)
        
        org_to_mod_act: dict = self.get_act_to_mod_act_dict(task=task)
        
        # get the actual parameterized actions and add their corresponding weights
        self.int_weight_dict = self._create_weight_dict(mod_action=modified_actions['robot'], task=task)

        # sort them according to their weights and then convert them in to addConst; reverse will sort the weights in descending order
        weight_dict = {k: v for k, v in sorted(self.int_weight_dict.items(), key=lambda item: item[1], reverse=True)}
        for action, w in weight_dict.items():
            weight_dict[action] = self.manager.addConst(int(w))
        
        sym_tr = DynWeightedPartitionedFrankaAbs(curr_vars=ts_curr_vars,
                                                 lbl_vars=ts_lbl_vars,
                                                 robot_action_vars=ts_robot_vars,
                                                 human_action_vars=ts_human_vars,
                                                 weight_dict=weight_dict,
                                                 int_weight_dict=self.int_weight_dict,
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
                                               mod_act_dict=org_to_mod_act)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)
        
        if print_facts:
            print(f"******************# of Edges in Franka Abstraction: {sym_tr.ecount}******************")
            print(f"******************# of States in the Original graph: {len(sym_tr.adj_map.keys())}******************")

        return sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars
    

    def build_bdd_bnd_abstraction_dynamic(self, max_human_int: int, draw_causal_graph: bool = False, print_facts: bool = True) \
         -> Tuple[BndDynamicFrankaTransitionSystem, List[BDD], List[BDD], List[BDD], List[BDD]]:
        """
         Main function to Build Two-player Game without edge weights and with bounded human intervention.
          The # of remainig Human interventions is added as a counter to each state.
        """
        task, domain, ts_curr_vars, ts_state_tuples, \
             ts_lbl_vars, ts_robot_vars, ts_human_vars, modified_actions, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                                                                   build_human_move=True,
                                                                                                                                   print_facts=print_facts)
        # we create human intervention boolean vars right  after the state vars 
        ts_hint_vars = self._create_symbolic_lbl_vars(state_lbls=list(range(max_human_int + 1)),
                                                      state_var_name='k',
                                                      add_flag=False)

        if print_facts:
            print(f"******************# of boolean Vars for Human Interventions: {len(ts_hint_vars)}******************")
        
        sym_tr = BndDynamicFrankaTransitionSystem(curr_vars=ts_curr_vars,
                                                  lbl_vars=ts_lbl_vars,
                                                  human_int_vars=ts_hint_vars,
                                                  robot_action_vars=ts_robot_vars,
                                                  human_action_vars=ts_human_vars,
                                                  task=task,
                                                  domain=domain,
                                                  ts_states=ts_state_tuples,
                                                  ts_state_map=self.pred_int_map,
                                                  max_human_int=max_human_int + 1,
                                                  ts_state_lbls=possible_lbls,
                                                  dfa_state_vars=self.dfa_x_list,
                                                  manager=self.manager,
                                                  sup_locs=self.sup_locs,
                                                  top_locs=self.top_locs,
                                                  modified_actions=modified_actions)
        
        start: float = time.time()
        sym_tr.create_transition_system_franka(boxes=boxes,
                                               add_exist_constr=True,
                                               verbose=self.verbose,
                                               plot=self.plot_ts,
                                               print_tr=False,
                                               debug=True)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)
        # sys.exit(-1)
        if print_facts:
            print(f"******************# of Edges in Franka Abstraction: {sym_tr.ecount}******************")
            print(f"******************# of States in the Original graph: {len(sym_tr.adj_map.keys())}******************")

        return sym_tr, ts_curr_vars, ts_robot_vars, ts_human_vars, ts_lbl_vars
    

    def build_bdd_symbolic_dfa(self, sym_tr_handle: Union[DynamicFrankaTransitionSystem, PartitionedFrankaTransitionSystem]) -> PartitionedDFA:
        """
         This function constructs the DFA in a partitioned fashion. We do not need two sets of variables to construct the transition relations.
        """
        if len(self.formulas) > 1:
            warnings.warn("Trying to construt Partitioned DFA representation for multiple Formulas. This functionality only works for Monolithic representation.")
            sys.exit(-1)

        start = time.time()
        # create TR corresponding to each DFA - dfa name is only used dumping graph 
        dfa_tr = PartitionedDFA(curr_states=self.dfa_x_list,
                                predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                sym_tr=sym_tr_handle,
                                manager=self.manager,
                                dfa=self.dfa_ltlf_handle,
                                ltlf_flag=self.ltlf_flag,
                                dfa_name='dfa_0')
        if self.ltlf_flag:
            dfa_tr.create_symbolic_ltlf_transition_system(verbose=self.verbose, plot=self.plot_dfa)
        else:
            raise NotImplementedError()
        stop = time.time()
        print("Time for Constructing the LTLF DFA: ", stop - start)

        return dfa_tr
    

    def build_add_symbolic_dfa(self, sym_tr_handle: DynWeightedPartitionedFrankaAbs) -> ADDPartitionedDFA:
        """
         A helper function to build a symbolic DFA given a formula from ADD Variables.
        """      
        if len(self.formulas) > 1:
            warnings.warn("Trying to construt Partitioned DFA representation for multiple Formulas. This functionality only works for Monolithic representation.")
            sys.exit(-1)

        start = time.time()
        # create TR corresponding to each DFA - dfa name is only used dumping graph 
        dfa_tr = ADDPartitionedDFA(curr_states=self.dfa_x_list,
                                   sym_tr=sym_tr_handle,
                                   manager=self.manager,
                                   dfa=self.dfa_ltlf_handle,
                                   ltlf_flag=self.ltlf_flag,
                                   dfa_name=f'dfa_0')
        
        if self.ltlf_flag:
            dfa_tr.create_symbolic_ltlf_transition_system(verbose=self.verbose, plot=self.plot_dfa)
        else:
            raise NotImplementedError()
        stop = time.time()
        print("Time for Constructing the LTLF DFA: ", stop - start)
        
        return dfa_tr
    

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
    

    def construct_best_effort_reach_strategies(self, adv_win_str: ADD, coop_win_str: ADD, adv_win_states: ADD, coop_win_states: ADD) -> ADD:
        """
         A function that constructs the best effort reachability strategies for the robot.

         For states (Winning Region) from where these exists a winning strategy, we retain them. For states (Pending Region), from where there is no winning strategy,
          we choose cooperative winning strategy. For the rest of the states (losing region), we choose all valid strategies.  
        """
        # TODO: Update the upper bound in the future to 2*max(edge_weight)*(|V| - 1). Need to check this bound.
        pending_states: BDD = ~adv_win_states.bddInterval(1, 10000) & coop_win_states.bddInterval(1, 10000)

        # retain the coop winning strategies for these states
        pending_states_str: ADD = pending_states.toADD() & coop_win_str

        # merge the straegies to construct BE strategies
        be_str: ADD = adv_win_str | pending_states_str

        return be_str
    

    def solve(self, verbose: bool = False, print_layers: bool = False, monolithic_tr: bool = False):
        """
         A function that call the winning strategy synthesis code and compute the set of winnign states and winning strategy for robot. 
        """
        print("**********************************************************************************************************")
        print("******************************************** TR: {approach} ***********************************************".format(approach='Monolithic' if monolithic_tr else 'Partitioned'))
        print("**********************************************************************************************************")
        start = time.time()
        if self.algorithm == 'qual':
            
            if isinstance(self.ts_handle, BndDynamicFrankaTransitionSystem):
                reachability_handle =  BndReachabilityGame(ts_handle=self.ts_handle,
                                                           dfa_handle=self.dfa_handle,
                                                           ts_curr_vars=self.ts_x_list,
                                                           dfa_curr_vars=self.dfa_x_list,
                                                           ts_obs_vars=self.ts_obs_list,
                                                           sys_act_vars=self.ts_robot_vars,
                                                           env_act_vars=self.ts_human_vars,
                                                           cudd_manager=self.manager)
            
            else:
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
            # sys.exit(-1)
            if win_str:
                reachability_handle.roll_out_strategy(transducer=win_str, verbose=True)
            
            # need to do this when testing 
            del reachability_handle.stra_list
            del reachability_handle.winning_states

        elif self.algorithm == 'quant-adv':
            min_max_handle = AdversarialGame(ts_handle=self.ts_handle,
                                             dfa_handle=self.dfa_handle,
                                             ts_curr_vars=self.ts_x_list,
                                             dfa_curr_vars=self.dfa_x_list,
                                             ts_obs_vars=self.ts_obs_list,
                                             sys_act_vars=self.ts_robot_vars,
                                             env_act_vars=self.ts_human_vars,
                                             cudd_manager=self.manager, 
                                             monolithic_tr=monolithic_tr)

            win_str: ADD = min_max_handle.solve(verbose=verbose, print_layers=print_layers)

            stop = time.time()
            print("Time for solving the game: ", stop - start)
            # sys.exit(-1)
            if min_max_handle.is_winning():
                min_max_handle.roll_out_strategy(strategy=win_str, verbose=True)
        
        elif self.algorithm == 'quant-coop':
            min_min_handle = CooperativeGame(ts_handle=self.ts_handle,
                                             dfa_handle=self.dfa_handle,
                                             ts_curr_vars=self.ts_x_list,
                                             dfa_curr_vars=self.dfa_x_list,
                                             ts_obs_vars=self.ts_obs_list,
                                             sys_act_vars=self.ts_robot_vars,
                                             env_act_vars=self.ts_human_vars,
                                             cudd_manager=self.manager,
                                             monolithic_tr=monolithic_tr)
            
            win_str: ADD = min_min_handle.solve(verbose=verbose, print_layers=print_layers)
            # sys.exit(-1)
            stop = time.time()
            print("Time for solving the game: ", stop - start)
            
            if min_min_handle.is_winning():
                min_min_handle.roll_out_strategy(strategy=win_str, verbose=True)
        
        elif self.algorithm == 'quant-be-reach':
            be_reach_handle = QuantitativeBestEffortReachSyn(ts_handle=self.ts_handle,
                                                             dfa_handle=self.dfa_handle,
                                                             ts_curr_vars=self.ts_x_list,
                                                             dfa_curr_vars=self.dfa_x_list,
                                                             ts_obs_vars=self.ts_obs_list,
                                                             sys_act_vars=self.ts_robot_vars,
                                                             env_act_vars=self.ts_human_vars,
                                                             cudd_manager=self.manager, 
                                                             monolithic_tr=monolithic_tr)
            
            be_str: ADD = be_reach_handle.solve(verbose=verbose, print_layers=print_layers)

            # BE always exists so we can always roll them out
            # count = 0
            # while count < 2:
                # print(f"Trial {count}")
            be_reach_handle.roll_out_strategy(verbose=True, intervene=True)
                # count += 1

        else:
            warnings.warn("Please enter either 'qual', 'quant-adv', or 'quant-coop' for Two player game strategy synthesis.")
            sys.exit(-1)
        