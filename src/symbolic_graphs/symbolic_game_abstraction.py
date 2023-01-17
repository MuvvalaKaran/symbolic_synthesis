import re
import sys
import time
import copy
import warnings
import graphviz as gv

from functools import reduce
from collections import defaultdict
from itertools import product
from typing import Tuple, List, Dict
from cudd import Cudd, BDD, ADD

from config import *

from bidict import bidict

from src.symbolic_graphs import PartitionedFrankaTransitionSystem


class DynamicFrankaTransitionSystem(PartitionedFrankaTransitionSystem):
    """
     A class that constructs symbolic Two-player Transition Relation. 
    """

    def __init__(self,
                 curr_vars: list,
                 lbl_vars: list,
                 robot_action_vars: list,
                 human_action_vars: list,
                 task, domain,
                 ts_state_map: dict,
                 ts_states: list,
                 manager: Cudd,
                 ts_state_lbls: list,
                 dfa_state_vars: List[BDD],
                 **kwargs):
        self.state_lbls = ts_state_lbls
        self.sym_vars_human: List[BDD] = human_action_vars
        self.sym_vars_robot: List[BDD] = robot_action_vars

        self.predicate_sym_map_human: bidict = {}
        self.predicate_sym_map_robot: bidict = {}

        super().__init__(curr_vars, lbl_vars, robot_action_vars, task, domain, ts_state_map, ts_states, manager, **kwargs)

        # index to determine where the state vars start 
        self.state_start_idx: int =  len(self.sym_vars_human) + len(self.sym_vars_robot) + len(dfa_state_vars)

        # create adj map. Useful when rolling out strategy with human intervention for sanity checking
        self.adj_map = defaultdict(lambda: defaultdict(lambda: {}))
        # edge counter
        self.ecount = 0

        self.state_cube = reduce(lambda x, y: x & y, self.sym_vars_curr)
        self.lbl_cube = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list])

        # for safety remove variable deprecated vars from parent class
        del self.sym_vars_action
        del self.predicate_sym_map_act


    def set_actions(self, **kwargs):
        """
         Override parent function to initialize valid set of actions to be a restrcited set
          of the original set of all parameterized actions.
        """
        acts = kwargs['modified_actions']['human'] + kwargs['modified_actions']['robot']
        self.actions: dict = [action for action in acts]
    
    
    def _initialize_sym_init_goal_states(self):
        """
        Initialize the inital states of the Transition system with their corresponding symbolic init state vairants.
        """
        self._create_sym_state_label_map(domain_lbls=self.state_lbls)
        init_tuple = self.get_tuple_from_state(self.init)
        goal_tuple = self.get_tuple_from_state(self.goal)
        
        self.sym_init_states = self.get_sym_state_from_tuple(init_tuple)
        self.sym_goal_states = None

        assert self.sym_init_states is not None, "Error extracting the Sym init state. FIX THIS!!!"

    
    def _initialize_bdds_for_actions(self):
        """
         This function intializes the mapping for both the robot and human actions to their corresponding
          Boolean formulas expressed using robot vars (o / output vairables/controllable vars) and
          human vars (i / input vairables/uncontrollable vars), respectively. 
        """
        for _pidx, var in enumerate([self.sym_vars_human, self.sym_vars_robot]):
            # create all combinations of 1-true and 0-false
            boolean_str = list(product([1, 0], repeat=len(var)))
            _node_int_map =  {}

            iter_count: int = 0
            for act in self.actions:
                if _pidx == 0  and 'human' in act:
                    _node_int_map[act] =  boolean_str[iter_count]
                    iter_count += 1
                elif _pidx == 1 and 'human' not in act:
                    _node_int_map[act] =  boolean_str[iter_count]
                    iter_count += 1

            assert len(boolean_str) >= len(_node_int_map), "FIX THIS: Looks like there are more Actions than boolean variables!"

            # loop over all the boolean strings and convert them respective bdd vars
            for _key, _value in _node_int_map.items():
                _act_val_list = []
                for _idx, _ele in enumerate(_value):
                    if _ele == 1:
                        _act_val_list.append(var[_idx])
                    else:
                        _act_val_list.append(~var[_idx])
                    
                    _bool_func_curr = reduce(lambda a, b: a & b, _act_val_list)

                    # update bidict accordingly
                    _node_int_map[_key] = _bool_func_curr
            
            if _pidx == 0:
                self.predicate_sym_map_human = bidict(_node_int_map)
            else:
                self.predicate_sym_map_robot = bidict(_node_int_map)
        
        self.initialize_sym_tr_action_list()
        
    
    def initialize_sym_tr_action_list(self):
        """
         Create a list of False BDDs for each paramterized action (Robot and Human) defined in the pddl domain file.
        """
        # initiate BDDs for all the action 
        action_idx_map = bidict()
        _actions = self.actions
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [[self.manager.bddZero() for _ in range(sum([len(listElem) for listElem in self.sym_vars_lbl]) + len(self.sym_vars_curr))] for _ in range(len(self.actions))]
    

    def get_mod_act_name(self, org_act_name: str) -> str:
        """
         A helper function that takes in as input the original name of the paramterized actions as parse by pyperplan, modifies it to
          the modified actions we manually create and returns that name
        """
        _loc_pattern = "[l|L][\d]+"
        _box_pattern = "[b|B][\d]+"

        if 'release' in org_act_name:
            return 'release'
        elif 'grasp' in org_act_name:
            return 'grasp'
        
        # transfer action are of type transfer l2 
        elif 'transfer' in org_act_name:
            locs: List[str] = re.findall(_loc_pattern, org_act_name)
            if 'else' in org_act_name:
                return f'transfer {locs[0]}'
            else:
                return f'transfer {locs[1]}'
            
        # transit action are of type transit b#
        elif 'transit' in org_act_name:
            _box_state: str = re.search(_box_pattern, org_act_name).group()
            return f'transit {_box_state}'

        elif 'human' in org_act_name:
            locs: List[str] = re.findall(_loc_pattern, org_act_name)
            _box_state: str = re.search(_box_pattern, org_act_name).group()

            return f'human-move {_box_state} {locs[1]}'
        else:
            warnings.warn("Could not look up the corresponding modified robot action name")
            sys.exit(-1)
    

    def get_state_tuple_from_sym_state(self, sym_state: BDD, sym_lbl_xcube_list: List[BDD]) -> tuple:
        """
         A function that loops over the entire sym state, extracts the corresponding predicates, looks up their index and returns the tuple
        """
        curr_state_name = self.predicate_sym_map_curr.inv[sym_state.existAbstract(self.lbl_cube)]
        curr_state_int = self.pred_int_map.get(curr_state_name)
        
        _lbl_list = []
        
        for idx in range(len(self.sym_vars_lbl)):
            # create a cube of the rest of the dfa vars
            exist_dfa_cube = self.manager.bddOne()
            for cube_idx, cube in enumerate(sym_lbl_xcube_list):
                if cube_idx != idx:
                    exist_dfa_cube = exist_dfa_cube & cube

            _lbl_dd = sym_state.existAbstract(self.state_cube & exist_dfa_cube)            
            _lbl_name = self.predicate_sym_map_lbl.inv[_lbl_dd]
            _lbl_int = self.pred_int_map.get(_lbl_name)
            
            assert _lbl_name is not None, "Couldn't convert LBL Cube to its corresponding State label. FIX THIS!!!"
            _lbl_list.append(_lbl_int)

        _lbl_list.extend([curr_state_int])

        return tuple(sorted(_lbl_list))


    def add_edge_to_action_tr(self,
                              curr_state_tuple: tuple,
                              next_state_tuple: tuple,
                              curr_state_sym: BDD,
                              nxt_state_sym: BDD,
                              curr_str_state: List[str]='',
                              next_str_state: List[str]='',
                              robot_action_name: str = '',
                              human_action_name: str = '',
                              valid_hact_list: BDD = None,
                              **kwargs) -> None:
        """
         While creating the TR for the two-player game, we add both human and robot action. We do encode every parameterized human move, i.e.,
          human-move b0 l6 l7 is simply added as human-move. This, restricts the no. of binary variables required to encode humans to two, i.e., 
          human intervene (human-move) and human not intervene (no-human-move)
        """
        if valid_hact_list is not None:
            assert isinstance(valid_hact_list, BDD), "Error Constructing TR Edges. Fix This!!!"
            no_human_move = valid_hact_list
        
        if human_action_name != '':
            assert robot_action_name != '', "Error While constructing Human Edge, FIX THIS!!!"

        # get the modified robot action name
        mod_raction_name: str = self.get_mod_act_name(org_act_name=robot_action_name)
        robot_move: BDD = self.predicate_sym_map_robot[mod_raction_name]
        

        if human_action_name != '':
            # get the modified human action name
            mod_haction_name: str = self.get_mod_act_name(org_act_name=human_action_name)
            _tr_idx: int = self.tr_action_idx_map.get(mod_haction_name)
            
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_str_state} --- {robot_action_name} & {human_action_name}---> {next_str_state}")
                
                self.mono_tr_bdd |= curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name]
        else:
            _tr_idx: int = self.tr_action_idx_map.get(mod_raction_name)
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & robot_move & no_human_move).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_str_state} ---{robot_action_name}---> {next_str_state}")

                self.mono_tr_bdd |= curr_state_sym & robot_move & no_human_move            

        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(nxt_state_sym.cube()):
            if var == 1 and self.manager.bddVar(_idx) in kwargs['prod_curr_list']:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                # if human intervenes then the edge looks like (true) & (human move b# l# l#)
                if human_action_name != '':
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name]
                    
                # if human does not intervene then the edge looks like (robot-action) & not(valid human moves)
                else:
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & no_human_move
                    
            
            elif var == 2 and self.manager.bddVar(_idx) in kwargs['prod_curr_list']:
                warnings.warn("Encountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)
        
        if human_action_name != '':
            if self.adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get('h') is None:
                self.adj_map[curr_state_tuple][mod_raction_name]['h'] = [next_state_tuple]
            else:
                self.adj_map[curr_state_tuple][mod_raction_name]['h'].append(next_state_tuple)
        else:
            if self.adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get('r') is not None:
                print("Error Computing Adj Dictionary, Fix this!!!")
                sys.exit(-1)
            
            self.adj_map[curr_state_tuple][mod_raction_name]['r'] = next_state_tuple
        
        # update edge count 
        self.ecount += 1
    

    def _check_exist_human_constraint(self, boxes: List[str], curr_state_lbl: tuple, human_action_name: str) -> bool:
        """
         A function that check if the destination location that the human is moving a block to is free or not.
        """
        _loc_pattern = "[l|L][\d]+"
        _box_pattern = "[b|B][\d]+"
        _loc_states: List[str] = re.findall(_loc_pattern, human_action_name)
        _box_state: str = re.search(_box_pattern, human_action_name).group()
        dloc = _loc_states[1]

        # human cannot move an object to TOP LOC
        if dloc in TOP_LOC:
            return False

        # if box1 is being manipulated, get the list of rest of boxes (that have to be grounded) at this instance
        # create predicates that say on b0 l1 (l1 is the destination in the action)
        on_preds = [self.pred_int_map[f'(on {bid} {dloc})'] for bid in boxes if bid != _box_state]
        
        if set(on_preds).intersection(set(curr_state_lbl)):
            return False

        return True
    

    def print_human_edge(self, curr_exp_states: List[str], hnext_exp_states: List[str], haction_name, **kwargs):
        """
         A helper function solely for printing an human intervention edge.
        """
        print(f"Adding Human edge: {curr_exp_states} -------{kwargs['robot_action_name']} & {haction_name}------> {hnext_exp_states}")
    

    def check_support_constraint(self, boxes: List[str], curr_state_lbl: tuple, human_action_name: str, **kwargs) -> bool:
        """
         Given the current human move check if the box being manipulated by the human is a support location or not. If yes, check if there is something in "top loc" or not.
        """
        _box_pattern = "[b|B][\d]+"
        _loc_pattern = "[l|L][\d]+"
        _chloc: List[str] = re.findall(_loc_pattern, human_action_name)[0]
        _box_state: str = re.search(_box_pattern, human_action_name).group()
        # check if the box is in support loc and has another box in top location, i.e,
        # if box1 is being manipulated, get the list of rest of boxes (that have to be grounded) at this instance

        if _chloc in SUP_LOC:  # if the human move is from current loc
            tmp_copy = copy.deepcopy(boxes)
            tmp_copy.remove(_box_state)

            # create predicates that say on b0 l1 (l1 is the destination in the action)
            on_preds = [self.pred_int_map[f'(on {bid} {tloc})'] for bid in tmp_copy for tloc in TOP_LOC]

            # check if a box exists on "top" in the curr state lbl or after the completion of the robot action
            if set(on_preds).intersection(set(curr_state_lbl)):
                return False
            
            # for Unbounded Abstraction
            if 'release' in kwargs.get('robot_action_name', ''):
                _dloc: str = re.search(_loc_pattern, kwargs['robot_action_name']).group()
                if _dloc in TOP_LOC:
                    return False

        return True


    def add_human_moves(self,
                        haction,
                        open_list: dict,
                        curr_state_tuple: tuple,
                        curr_exp_states: List[str],
                        curr_state_sym: BDD,
                        curr_rob_conf: str,
                        layer: int,
                        boxes: List[str],
                        verbose: bool = False,
                        **kwargs) -> bool:
        """
         A function that loops over all the human moves, check if the preconditions are met, if yes then compute the next state and add it to the TR.  

         Note: When the human does intervene, we add the effects of the robot action as well as the human action. 
         Robot add effect will (should) not clash with human's del effect
        """
        _box_pattern = "[b|B][\d]+"
        _loc_pattern = "[l|L][\d]+"
        # we do not allow the human move the obj the robot is currently grasping
        if 'to-obj' in curr_rob_conf:
            _box_state: str = re.search(_box_pattern, curr_rob_conf).group()
            _hbox_state: str = re.search(_box_pattern, haction.name).group()

            if _box_state == _hbox_state:
                return False
        
        # Check release only when you are constructing Bounded Abstraction
        elif kwargs.get('robot_action_name') is not None and 'release' in kwargs['robot_action_name']:
            _dloc: str = re.search(_loc_pattern, kwargs['robot_action_name']).group()
            _hloc: str = re.findall(_loc_pattern, haction.name)[1]

            if _dloc == _hloc:
                return False
            
        # check if the destination loc is free or not
        d_loc_available: bool = self._check_exist_human_constraint(boxes=boxes,
                                                                   curr_state_lbl=curr_state_tuple,
                                                                   human_action_name=haction.name)

        if d_loc_available:
            valid_move: bool = self.check_support_constraint(boxes=boxes,
                                                             curr_state_lbl=curr_state_tuple,
                                                             human_action_name=haction.name,
                                                             **kwargs)
            if not valid_move:
                return False

            # for Unvounded Abstraction we allow the robot to evolve 
            if kwargs.get('next_exp_states') is not None:
                next_exp_state = list(haction.apply(state=frozenset(kwargs['next_exp_states'])))

                if 'transit' in kwargs['robot_action_name']:
                    _hcloc: str = re.findall(_loc_pattern, haction.name)[0]
                    try:
                        # fails when transiting from else loc to l#
                        _dloc: str = re.findall(_loc_pattern, kwargs['robot_action_name'])[1]
                    except:
                        _dloc: str = re.findall(_loc_pattern, kwargs['robot_action_name'])[0]
                    
                    _box_state: str = re.search(_box_pattern, kwargs['robot_action_name']).group()

                    if _hcloc == _dloc:
                        # remove to-obj prediacte
                        next_exp_state = list(set(next_exp_state) - set([f'(to-obj {_box_state})']))
                        next_exp_state = next_exp_state + [f'(ready {_dloc})']

            else:
                # look up its corresponding formula - for Bounded Human Intervention
                next_exp_state = list(haction.apply(state=frozenset(curr_exp_states)))
            
            next_sym_state: BDD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)


            if verbose:
                self.print_human_edge(curr_exp_states=list(curr_exp_states),
                                      hnext_exp_states=next_exp_state,
                                      haction_name=haction.name,
                                      **kwargs)

            # add The edge to its corresponding action
            self.add_edge_to_action_tr(curr_state_tuple=curr_state_tuple,
                                       curr_state_sym=curr_state_sym,
                                       next_state_tuple=next_exp_state,
                                       nxt_state_sym=next_sym_state,
                                       human_action_name=haction.name,
                                       **kwargs)

            # store the image in the next bucket
            if kwargs.get('curr_hint') is not None:
                # add state & remaning human intervention to the next bucket
                curr_hint: int = kwargs['curr_hint']
                open_list[layer + 1] |= next_sym_state & self.predicate_sym_map_hint[curr_hint - 1]
            else:
                # only add state to the next bucket
                open_list[layer + 1] |= next_sym_state

            # get their corresponding lbls 
            self.sym_state_labels |= next_sym_state 
    
            return True
        
        return False
    

    def create_human_edges(self,
                           human_actions: list,
                           curr_sym_state: BDD,
                           curr_state_tuple: tuple,
                           open_list: List[BDD],
                           boxes: List[str],
                           layer: int,
                           prod_curr_list: List[BDD],
                           verbose: bool = False,
                           **kwargs) -> List:
        """
         A function that loops through all the human edges and add the valid moves to the symbolic TR.
         We then return the list of valid TS actions to create the valid robot edges from the symbolic state.  
        """
        curr_exp_states = frozenset(self.get_state_from_tuple(curr_state_tuple))
        # get valid pres from current state tuple
        pre_robot_conf = self.get_conds_from_state(curr_state_tuple, only_robot_conf=True)
        curr_rstate = self.get_state_from_tuple(pre_robot_conf)

        assert len(curr_rstate) == 1, "Error computing Robot Configuration. Fix This!!!"

        env_edge_acts = []
        # compute the image of the TS states
        for action in human_actions:
            valid_act: bool = False
            if action.applicable(state=curr_exp_states):
                # Add human moves
                valid_act =  self.add_human_moves(haction=action,
                                                  open_list=open_list,
                                                  curr_state_tuple=curr_state_tuple,
                                                  curr_exp_states=curr_exp_states,
                                                  curr_state_sym=curr_sym_state,
                                                  curr_rob_conf=curr_rstate[0],
                                                  layer=layer,
                                                  boxes=boxes,
                                                  verbose=verbose,
                                                  prod_curr_list=prod_curr_list,
                                                  **kwargs)
            if valid_act:
                env_edge_acts.append(action.name)

        return env_edge_acts    
    

    def create_transition_system_franka(self,
                                        boxes: List[str],
                                        add_exist_constr: bool = True,
                                        verbose: bool = False,
                                        plot: bool = False,
                                        **kwargs):
        """
         This method overrides the parent method. For every robot action, we loop over all the human actions.
        """
        open_list = defaultdict(lambda: self.manager.bddZero())

        closed = self.manager.bddZero()

        init_state_sym = self.sym_init_states
        self.sym_state_labels |= init_state_sym
        
        layer = 0

        # creating monolithinc tr bdd to keep track all the transition we are creating to detect nondeterminism
        if 'debug' in kwargs:
            self.mono_tr_bdd = self.manager.bddZero() 

        # no need to check if other boxes are placed at the destination loc during transfer and release as there is only one object
        if len(boxes) == 1:
            add_exist_constr = False
        
        prod_curr_list = [lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list]
        prod_curr_list.extend([*self.sym_vars_curr])

        open_list[layer] |= init_state_sym

        # create sym_lbl_cube list
        sym_lbl_xcube_list = [] 
        for vars_list in self.sym_vars_lbl:
            sym_lbl_xcube_list.append(reduce(lambda x, y: x & y, vars_list))
        
        # segregate actions into human and robot actions
        _seg_actions = {'human': [], 'robot': []}
        for act in self.task.operators:
            if 'human' in act.name:
                _seg_actions['human'].append(act)
            else:
                _seg_actions['robot'].append(act)


        while not open_list[layer].isZero():
            # remove all states that have been explored
            open_list[layer] = open_list[layer] & ~closed

            # If unexpanded states exist ...
            if not open_list[layer].isZero():
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer]

                # if verbose:
                print(f"******************************* Layer: {layer}*******************************")
                # get all the states
                sym_state = self._convert_state_lbl_cube_to_func(dd_func= open_list[layer], prod_curr_list=prod_curr_list)
                for state in sym_state:
                    curr_state_tuple = self.get_state_tuple_from_sym_state(sym_state=state, sym_lbl_xcube_list=sym_lbl_xcube_list)
                    curr_exp_states = frozenset(self.get_state_from_tuple(curr_state_tuple))
                    for raction in  _seg_actions['robot']:
                        # set action feasbility flag to True - used during transfer and release action to check the des loc is empty
                        action_feas: bool = True

                        if raction.applicable(state=curr_exp_states):
                            # add existential constraints to transfer and relase action
                            if add_exist_constr and (('transfer' in raction.name) or ('release' in raction.name)):
                                action_feas = self._check_exist_constraint(boxes=boxes,
                                                                           curr_state_lbl=curr_state_tuple,
                                                                           action_name=raction.name)
                            
                            if not action_feas:
                                continue

                            next_exp_state = list(raction.apply(state=frozenset(curr_exp_states)))
                            next_sym_state: BDD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)

                            # create human edges
                            valid_human_edges = self.create_human_edges(human_actions=_seg_actions['human'],
                                                                        curr_sym_state=state,
                                                                        curr_state_tuple=curr_state_tuple,
                                                                        sym_lbl_cubes=sym_lbl_xcube_list,
                                                                        open_list=open_list,
                                                                        layer=layer,
                                                                        boxes=boxes,
                                                                        prod_curr_list=prod_curr_list,
                                                                        verbose=verbose,
                                                                        robot_action_name=raction.name,
                                                                        next_exp_states=next_exp_state,
                                                                        **kwargs)

                            # if there are any valid human edges from curr state
                            if len(valid_human_edges) > 0:
                                valid_hact: List[BDD] = [self.predicate_sym_map_human[self.get_mod_act_name(ha)] for ha in valid_human_edges]
                                no_human_move_edge: BDD = ~(reduce(lambda x, y: x | y, valid_hact))
                            else:
                                no_human_move_edge: BDD = self.manager.bddOne()
                    
                            # create robot edges
                            self.add_edge_to_action_tr(robot_action_name=raction.name,
                                                        curr_state_tuple=curr_state_tuple,
                                                        next_state_tuple=next_exp_state,
                                                        curr_state_sym=state,
                                                        nxt_state_sym=next_sym_state,
                                                        valid_hact_list=no_human_move_edge,
                                                        prod_curr_list=prod_curr_list,
                                                        **kwargs)
                            if verbose:
                                print(f"Adding Robot edge: {list(curr_exp_states)} -------{raction.name}------> {next_exp_state}")

                            # get their corresponding lbls 
                            self.sym_state_labels |= next_sym_state 

                            open_list[layer + 1] |= next_sym_state
                
                layer += 1
        
        if kwargs['print_tr'] is True:
            self._print_plot_tr(plot=plot)

    

class BndDynamicFrankaTransitionSystem(DynamicFrankaTransitionSystem):
    """
     A class that constructs symbolic Two-player Transition Relation with bounded Human interventions
    """

    def __init__(self,
                 curr_vars: list,
                 lbl_vars: list,
                 human_int_vars: list,
                 robot_action_vars: list,
                 human_action_vars: list,
                 task, domain,
                 ts_state_map: dict,
                 ts_states: list,
                 max_human_int: int,
                 manager: Cudd,
                 ts_state_lbls: list,
                 dfa_state_vars: List[BDD],
                 **kwargs):
        self.sym_vars_hint: List[BDD] = human_int_vars
        super().__init__(curr_vars, lbl_vars, robot_action_vars, human_action_vars, task, domain, ts_state_map, ts_states, manager, ts_state_lbls, dfa_state_vars, **kwargs)
        
        self.max_hint: int = max_human_int

        self.predicate_sym_map_hint: bidict = {}

        self._initialize_bdd_for_human_int()

        # store the bdd associated with each state vars in this list. The index corresonds to its number
        self.tr_state_bdds = [self.manager.bddZero() for _ in range(sum([len(listElem) for listElem in self.sym_vars_lbl]) + len(self.sym_vars_curr) + len(self.sym_vars_hint))]

        # create adj map. Useful when rolling out strategy with human intervention for sanity checking
        self.adj_map = defaultdict(lambda: defaultdict(lambda : {}))

        self.hint_cube = reduce(lambda x, y: x & y, self.sym_vars_hint)


    def initialize_sym_tr_action_list(self):
        """
         Override the parent function as we have additional boolean variables corresponding
          to Human intervention and thus the length of the state bdd is different.
        """
        # initiate BDDs for all the action 
        action_idx_map = bidict()
        _actions = self.actions
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [[self.manager.bddZero() for _ in range(sum([len(listElem) for listElem in self.sym_vars_lbl]) + len(self.sym_vars_curr) + len(self.sym_vars_hint))] for _ in range(len(self.actions))]


    def _initialize_bdd_for_human_int(self):
        """
         This function initializes bdd that represents the number of human interventions left at each state. 
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_hint)))
        _node_int_map = bidict({_hint: boolean_str[_hint] for _hint in range(self.max_hint)})

        assert len(boolean_str) >= len(_node_int_map), \
             "FIX THIS: Looks like there are more human interventions than it's corresponding boolean variables!"
        
        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map.items():
            _val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _val_list.append(self.sym_vars_hint[_idx])
                else:
                    _val_list.append(~self.sym_vars_hint[_idx])
            
            _bool_func_curr = reduce(lambda a, b: a & b, _val_list)

            # update bidict accordingly
            _node_int_map[_key] = _bool_func_curr
        
        self.predicate_sym_map_hint = bidict(_node_int_map)


    def add_edge_to_action_tr(self,
                              curr_state_tuple: tuple,
                              next_state_tuple: tuple,
                              curr_state_sym: BDD,
                              nxt_state_sym: BDD,
                              curr_str_state: List[str]='',
                              next_str_state: List[str]='',
                              robot_action_name: str = '',
                              human_action_name: str = '',
                              valid_hact_list: BDD = None,
                              **kwargs) -> None:
        if valid_hact_list is None:
            valid_hact_list = []

        else:
            assert isinstance(valid_hact_list, BDD), "Error Constructing TR Edges. Fix This!!!"
            no_human_move = valid_hact_list

        if human_action_name != '':
            assert robot_action_name == '', "Error While constructing Human Edge, FIX THIS!!!"
            robot_move = self.manager.bddOne()

        elif robot_action_name != '':
            # get the modified robot action name
            mod_raction_name: str = self.get_mod_act_name(org_act_name=robot_action_name)
            robot_move: BDD = self.predicate_sym_map_robot[mod_raction_name]

        curr_hint: int = kwargs['curr_hint']
        
        if human_action_name != '':
            # get the modified human action name
            mod_haction_name: str = self.get_mod_act_name(org_act_name=human_action_name)
            nxt_state_sym = nxt_state_sym & self.predicate_sym_map_hint[curr_hint - 1]
            _tr_idx: int = self.tr_action_idx_map.get(mod_haction_name)

            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name] & self.predicate_sym_map_hint[curr_hint]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_str_state} ---{human_action_name}---> {next_str_state}")

                self.mono_tr_bdd |= curr_state_sym & self.predicate_sym_map_human[mod_haction_name] & self.predicate_sym_map_hint[curr_hint]
        else:
            nxt_state_sym = nxt_state_sym & self.predicate_sym_map_hint[curr_hint]
            _tr_idx: int = self.tr_action_idx_map.get(mod_raction_name)

            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd &  curr_state_sym & robot_move & no_human_move & self.predicate_sym_map_hint[curr_hint]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Robot Action: {curr_str_state}[{curr_hint}] ---{robot_action_name}---> {next_str_state}[{curr_hint}]")
                
                self.mono_tr_bdd |= curr_state_sym & robot_move & no_human_move & self.predicate_sym_map_hint[curr_hint]             


        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(nxt_state_sym.cube()):
            if var == 1 and self.manager.bddVar(_idx) in kwargs['prod_curr_list']:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                # if human intervenes then the edge looks like (true) & (human move b# l# l#)
                if human_action_name != '':
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name] & self.predicate_sym_map_hint[curr_hint]
                # if human does not intervene then the edge looks like (robot-action) & not(valid human moves)
                else:
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & no_human_move & self.predicate_sym_map_hint[curr_hint]

            
            elif var == 2 and self.manager.bddVar(_idx) in kwargs['prod_curr_list']:
                warnings.warn("Encountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)
        
        if human_action_name != '':
            if self.adj_map.get(curr_state_tuple, {}).get(curr_hint, {}).get('h') is None:
                self.adj_map[curr_state_tuple][curr_hint]['h'] = [next_state_tuple]
            else:
                self.adj_map[curr_state_tuple][curr_hint]['h'].append(next_state_tuple)

        else:
            if self.adj_map.get(curr_state_tuple, {}).get(curr_hint, {}).get(mod_raction_name) is not None:
                print("Error Computing Adj Dictionary, Fix this!!!")
                sys.exit(-1)

            self.adj_map[curr_state_tuple][curr_hint][mod_raction_name] = next_state_tuple
        
        # update edge count 
        self.ecount += 1
    

    def create_robot_edges(self,
                           robot_actions: list,
                           curr_sym_state: BDD,
                           curr_state_tuple: tuple,
                           open_list: List[BDD],
                           boxes: List[str],
                           layer: int,
                           human_edge: BDD,
                           add_exist_constr: bool,
                           prod_curr_list: List[BDD],
                           verbose: bool = False,
                           **kwargs):
        """
         A function that loop create the valid robot edges given the current state and set the of valid edges from the current state.
        """
        curr_exp_states = frozenset(self.get_state_from_tuple(curr_state_tuple))
        # compute the image of the TS states
        for action in robot_actions:
            # set action feasbility flag to True - used during transfer and release action to check the des loc is empty
            action_feas: bool = True

            if action.applicable(state=curr_exp_states):
                # add existential constraints to transfer and relase action
                if add_exist_constr and (('transfer' in action.name) or ('release' in action.name)):
                    action_feas = self._check_exist_constraint(boxes=boxes,
                                                               curr_state_lbl=curr_state_tuple,
                                                               action_name=action.name)
                
                if not action_feas:
                    continue

                next_exp_state = list(action.apply(state=frozenset(curr_exp_states)))
                next_sym_state: BDD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)

                # add The edge to its corresponding action
                self.add_edge_to_action_tr(robot_action_name=action.name,
                                           curr_state_tuple=curr_state_tuple,
                                           next_state_tuple=next_exp_state,
                                           curr_state_sym=curr_sym_state,
                                           nxt_state_sym=next_sym_state,
                                           valid_hact_list=human_edge,
                                           prod_curr_list=prod_curr_list,
                                           **kwargs)
                
                if verbose:
                    # if kwargs.get('curr_hint') is not None:
                    print(f"Adding Robot edge: {list(curr_exp_states)}[{kwargs['curr_hint']}] -------{action.name}------> {next_exp_state}[{kwargs['curr_hint']}]")
                    # else:
                    #     print(f"Adding Robot edge: {list(curr_exp_states)} -------{action.name}------> {next_exp_state}")

                # get their corresponding lbls 
                self.sym_state_labels |= next_sym_state 

                # store the image in the next bucket
                # if kwargs.get('curr_hint') is not None:
                curr_hint: int = kwargs['curr_hint']
                open_list[layer + 1] |= next_sym_state & self.predicate_sym_map_hint[curr_hint]
                # else:
                #     open_list[layer + 1] |= next_sym_state
    

    def get_state_tuple_from_sym_state(self, sym_state: BDD, sym_lbl_xcube_list: List[BDD]) -> Tuple[tuple, int]:
        """
         A function that loops over the entire sym state, extracts the corresponding predicates, looks up their index and returns the tuple
        """
        curr_state_name = self.predicate_sym_map_curr.inv[sym_state.existAbstract(self.hint_cube & self.lbl_cube)]
        curr_state_int = self.pred_int_map.get(curr_state_name)
        curr_hint: int = self.predicate_sym_map_hint.inv[sym_state.existAbstract(self.state_cube & self.lbl_cube)]
        
        _lbl_list = []
        
        for idx in range(len(self.sym_vars_lbl)):
            # create a cube of the rest of the dfa vars
            exist_dfa_cube = self.manager.bddOne()
            for cube_idx, cube in enumerate(sym_lbl_xcube_list):
                if cube_idx != idx:
                    exist_dfa_cube = exist_dfa_cube & cube

            _lbl_dd = sym_state.existAbstract(self.state_cube & self.hint_cube & exist_dfa_cube)            
            _lbl_name = self.predicate_sym_map_lbl.inv[_lbl_dd]
            _lbl_int = self.pred_int_map.get(_lbl_name)
            
            assert _lbl_name is not None, "Couldn't convert LBL Cube to its corresponding State label. FIX THIS!!!"
            _lbl_list.append(_lbl_int)

        _lbl_list.extend([curr_state_int])

        return tuple(sorted(_lbl_list)), curr_hint
    

    def create_human_edges(self,
                           human_actions: list,
                           curr_sym_state: BDD,
                           curr_state_tuple: tuple,
                           open_list: List[BDD],
                           boxes: List[str],
                           layer: int,
                           prod_curr_list: List[BDD],
                           verbose: bool = False,
                           **kwargs) -> List:
        """
         A function that loops through all the human edges and add the valid moves to the symbolic TR.
         We then return the list of valid TS actions to create the valid robot edges from the symbolic state.  
        """
        # curr_state_tuple, curr_hint = self.get_state_tuple_from_sym_state(sym_state=curr_sym_state, sym_lbl_xcube_list=sym_lbl_cubes)
        curr_exp_states = frozenset(self.get_state_from_tuple(curr_state_tuple))
        # get valid pres from current state tuple
        pre_robot_conf = self.get_conds_from_state(curr_state_tuple, only_robot_conf=True)
        curr_rstate = self.get_state_from_tuple(pre_robot_conf)

        assert len(curr_rstate) == 1, "Error computing Robot Configuration. Fix This!!!"

        env_edge_acts = []
        # compute the image of the TS states
        for action in human_actions:
            valid_act: bool = False
            if action.applicable(state=curr_exp_states):
                # Add human moves
                if kwargs['curr_hint'] > 0:
                    valid_act =  self.add_human_moves(haction=action,
                                                      open_list=open_list,
                                                      curr_state_tuple=curr_state_tuple,
                                                      curr_exp_states=curr_exp_states,
                                                      curr_state_sym=curr_sym_state,
                                                      curr_rob_conf=curr_rstate[0],
                                                      layer=layer,
                                                      boxes=boxes,
                                                      verbose=verbose,
                                                      prod_curr_list=prod_curr_list,
                                                      **kwargs)
            if valid_act:
                env_edge_acts.append(action.name)

        return env_edge_acts

    
    def print_human_edge(self, curr_exp_states: List[str], hnext_exp_states: List[BDD], haction_name, **kwargs):
        """
         Override base printing method to include remaining human intervention associated with each state.
        """
        print(f"Adding Human edge: {curr_exp_states}[{kwargs['curr_hint']}] -------{haction_name}------> {hnext_exp_states}[{kwargs['curr_hint'] - 1}]")


    def create_transition_system_franka(self,
                                        boxes: List[str],
                                        add_exist_constr: bool = True,
                                        verbose: bool = False,
                                        plot: bool = False,
                                        **kwargs):
        """
         This method overrides DynamicFrankaTransitionSystem method where we do not encode the No. of human actions remaining explicitly in each node.

         Thus, from a state (S, K), if the human intervenes, then we transit to a state with (S', K-1) where S -> S' is a valid transtion in the TS and K is the # of human interventions remaining. 

         Note: When K=0; the human can not intervene anymore. 
               Pass 'debug' (kwarg) as True for sanity checking. The Compositional Approach as outlined
               by Zhu et al. (LTLF Syft) does not work when you have nondeterminism, i.e., for same (State, Action) pair, you cannot evolve to two distinct successors.
               By setting debug to True, we keep track of all the edge created during the Abstarction Construction process.  
        """
        open_list = defaultdict(lambda: self.manager.bddZero())

        closed = self.manager.bddZero()

        init_state_sym = self.sym_init_states

        # get the state lbls and create state and state lbl mappinng
        init_hint: BDD = self.predicate_sym_map_hint[self.max_hint - 1]

        # update the init state with hint
        self.sym_init_states = self.sym_init_states & init_hint
        
        # each states consists of boolean Vars corresponding to: S, LBL; K
        self.sym_state_labels |= init_state_sym
     
        # human int cube 
        layer = 0

        # creating monolithinc tr bdd to keep track all the transition we are creating to detect nondeterminism
        if 'debug' in kwargs:
            self.mono_tr_bdd = self.manager.bddZero() 

        # no need to check if other boxes are placed at the destination loc during transfer and release as there is only one object
        if len(boxes) == 1:
            add_exist_constr = False
        
        prod_curr_list = [lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list]
        prod_curr_list.extend([*self.sym_vars_curr, *self.sym_vars_hint])

        # a state is fully defined with 
        open_list[layer] |= init_state_sym & init_hint

        # create sym_lbl_cube list
        sym_lbl_xcube_list = [] 
        for vars_list in self.sym_vars_lbl:
            sym_lbl_xcube_list.append(reduce(lambda x, y: x & y, vars_list))
        
        # segregate actions into human and robot actions
        _seg_actions = {'human': [], 'robot': []}
        for act in self.task.operators:
            if 'human' in act.name:
                _seg_actions['human'].append(act)
            else:
                _seg_actions['robot'].append(act)

        while not open_list[layer].isZero():
            # remove all states that have been explored
            open_list[layer] = open_list[layer] & ~closed

            # If unexpanded states exist ...
            if not open_list[layer].isZero():
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer]

                # if verbose:
                print(f"******************************* Layer: {layer}*******************************")

                # get all the states and their corresponding remaining human intervention
                # sta = time.time()
                sym_state = self._convert_state_lbl_cube_to_func(dd_func=open_list[layer], prod_curr_list=prod_curr_list)
                # sto = time.time()
                # print("Time spent computing the cubes: ", sto -sta)
                # sta = time.time()
                for state in sym_state:
                    curr_state_tuple, curr_hint = self.get_state_tuple_from_sym_state(sym_state=state, sym_lbl_xcube_list=sym_lbl_xcube_list)
                    # create human edges
                    valid_human_edges = self.create_human_edges(human_actions=_seg_actions['human'],
                                                                curr_sym_state=state,
                                                                curr_state_tuple=curr_state_tuple,
                                                                sym_lbl_cubes=sym_lbl_xcube_list,
                                                                open_list=open_list,
                                                                layer=layer,
                                                                boxes=boxes,
                                                                prod_curr_list=prod_curr_list,
                                                                verbose=verbose,
                                                                curr_hint=curr_hint,
                                                                **kwargs)
                    # if there are any valid human edges from curr state
                    if len(valid_human_edges) > 0:
                        valid_hact: List[BDD] = [self.predicate_sym_map_human[self.get_mod_act_name(ha)] for ha in valid_human_edges]
                        no_human_move_edge: BDD = ~(reduce(lambda x, y: x | y, valid_hact))
                    else:
                        no_human_move_edge: BDD = self.manager.bddOne()
                    
                    
                    # create robot edges
                    self.create_robot_edges(robot_actions=_seg_actions['robot'],
                                            curr_sym_state=state,
                                            curr_state_tuple=curr_state_tuple,
                                            sym_lbl_cubes=sym_lbl_xcube_list,
                                            open_list=open_list,
                                            layer=layer,
                                            boxes=boxes,
                                            human_edge=no_human_move_edge,
                                            add_exist_constr=add_exist_constr,
                                            prod_curr_list=prod_curr_list,
                                            verbose=verbose,
                                            curr_hint=curr_hint,
                                            **kwargs)
                    
                # sto = time.time()
                # print("Time spent costructing the Edges: ", sto -sta)
                layer += 1
        
        if kwargs['print_tr'] is True:
            self._print_plot_tr(plot=plot)