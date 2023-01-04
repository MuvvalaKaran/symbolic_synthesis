import re
import sys
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
                 manager: Cudd):
        self.sym_vars_human: List[BDD] = human_action_vars
        self.sym_vars_robot: List[BDD] = robot_action_vars

        self.predicate_sym_map_human: bidict = {}
        self.predicate_sym_map_robot: bidict = {}

        super().__init__(curr_vars, lbl_vars, robot_action_vars, task, domain, ts_state_map, ts_states, manager)

        # index to determine where the state vars start 
        self.state_start_idx: int = len(self.sym_vars_lbl) + len(self.sym_vars_human) + len(self.sym_vars_robot)

        # create adj map. Useful when rolling out strategy with human intervention for sanity checking
        self.adj_map = defaultdict(lambda: defaultdict(lambda : {'h': [], 'r': []}))
        # edge counter
        self.ecount = 0

        # for safety remove variable deprecated vars from parent class
        del self.sym_vars_action
        del self.predicate_sym_map_act


    
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
            for state in self.actions:
                if _pidx == 0  and 'human' in state:
                    _node_int_map[state] =  boolean_str[iter_count]
                    iter_count += 1
                elif _pidx == 1 and 'human' not in state:
                    _node_int_map[state] =  boolean_str[iter_count]
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
        self.sym_tr_actions = [[self.manager.bddZero() for _ in range(len(self.sym_vars_curr))] for _ in range(len(self.actions))]
        

    def add_edge_to_action_tr(self,
                              robot_action_name: str,
                              curr_state_tuple: tuple,
                              next_state_tuple: tuple,
                              human_action_name: str = '',
                              valid_hact_list: List[str] = None,
                              **kwargs) -> None:
        """
         While creating the TR for the two-player game, we add both human and robot action. We do encode every parameterized human move, i.e.,
          human-move b0 l6 l7 is simply added as human-move. This, restricts the no. of binary variables required to encode humans to two, i.e., 
          human intervene (human-move) and human not intervene (no-human-move)
        """
        # design pattern to avoind mutables as default args. 
        if valid_hact_list is None:
            valid_hact_list = []
        elif isinstance(valid_hact_list, list):
            # if there are any valid human edges from curr state
            if len(valid_hact_list) > 0:
                valid_hact: List[BDD] = [self.predicate_sym_map_human[ha] for ha in valid_hact_list]
                no_human_move: BDD = ~(reduce(lambda x, y: x | y, valid_hact))
            else:
                no_human_move: BDD = self.manager.bddOne()

        else:
            warnings.warn("Invalid Default args type when constructing the Symbolic Franka Abstraction. FIX THIS!!!")
            sys.exit(-1)
        
        curr_state_sym: BDD = self.predicate_sym_map_curr[curr_state_tuple]
        nxt_state_sym: BDD = self.predicate_sym_map_curr[next_state_tuple]

        if human_action_name != '':
            _tr_idx: int = self.tr_action_idx_map.get(human_action_name)
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & self.predicate_sym_map_human[human_action_name]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {self.get_state_from_tuple(curr_state_tuple)} ---{human_action_name}---> {self.get_state_from_tuple(next_state_tuple)}")
                
                self.mono_tr_bdd |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & self.predicate_sym_map_human[human_action_name]
        else:
            _tr_idx: int = self.tr_action_idx_map.get(robot_action_name)
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd &  curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & no_human_move).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Robot Action: {self.get_state_from_tuple(curr_state_tuple)} ---{robot_action_name}---> {self.get_state_from_tuple(next_state_tuple)}")
                
                self.mono_tr_bdd |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & no_human_move            

        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(nxt_state_sym.cube()):
            if var == 1 and self.manager.bddVar(_idx) in self.sym_vars_curr:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                # if human intervenes then the edge looks like (robot-action) & (human move b# l# l#)
                if human_action_name != '':
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & self.predicate_sym_map_human[human_action_name]
                    # self.tr_state_bdds[_state_idx] |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & self.predicate_sym_map_human[human_action_name] 
                # if human does not intervene then the edge looks like (robot-action) & not(valid human moves)
                else:
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & no_human_move
                    # self.tr_state_bdds[_state_idx] |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & no_human_move
            
            elif var == 2 and self.manager.bddVar(_idx) in self.sym_vars_curr:
                warnings.warn("Encountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)
        
        if human_action_name != '':
            self.adj_map[curr_state_tuple][robot_action_name]['h'].append(next_state_tuple)
        else:
            self.adj_map[curr_state_tuple][robot_action_name]['r'].append(next_state_tuple)
        
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
        tmp_copy = copy.deepcopy(boxes)
        tmp_copy.remove(_box_state)

        # create predicates that say on b0 l1 (l1 is the destination in the action)
        on_preds = [self.pred_int_map[f'(on {bid} {dloc})'] for bid in tmp_copy]
        
        if set(on_preds).intersection(set(curr_state_lbl)):
            return False

        return True
    
    def print_human_edge(self, curr_state_tuple: tuple, hnext_tuple: tuple, robot_action_name: str, haction, **kwargs):
        """
         A helper function solely for printing an human intervention edge.
        """
        cstate = self.get_state_from_tuple(state_tuple=tuple(curr_state_tuple))
        nstate = self.get_state_from_tuple(state_tuple=hnext_tuple)
        print(f"Adding Human edge: {cstate} -------{robot_action_name} & {haction.name}------> {nstate}")
    

    def check_support_constraint(self, boxes: List[str], curr_state_lbl: tuple, human_action_name: str, robot_action_name: str) -> bool:
        """
         Given the current human move check if the box being manipulated by the human is a support location or not. If yes, heck if there is something in "top loc" or not.
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
            
            if 'release' in robot_action_name:
                _dloc: str = re.search(_loc_pattern, robot_action_name).group()
                if _dloc in TOP_LOC:
                    return False

        return True


    def add_human_moves(self,
                        robot_action_name: str,
                        open_list: dict,
                        curr_state_tuple: tuple,
                        robot_nxt_tuple: tuple,
                        layer: int,
                        boxes: List[str],
                        verbose: bool = False,
                        **kwargs) -> List[str]:
        """
         A function that loops over all the human moves, check if the preconditions are met, if yes then compute the next state and add it to the TR.  

         Note: When the human does intervene, we add the effects of the robot action as well as the human action. 
         Robot add effect will (should) not clash with human's del effect
        """
        # keep track of valid actions the human took
        _hact_list = []
        for haction in self.task.operators:
            if 'human' not in haction.name:
                continue
            
            # check if the preconditions of human action are satisfied or not
            _pre_tuple = self.get_tuple_from_state(haction.preconditions)  # cardinality of human move's precondition is always 1
            _intersect: bool = set(_pre_tuple).issubset(curr_state_tuple)
            _box_pattern = "[b|B][\d]+"
            _loc_pattern = "[l|L][\d]+"

            # we do not allow the human move the obj the robot is currently grasping
            if 'grasp' in robot_action_name:
                _box_state: str = re.search(_box_pattern, robot_action_name).group()
                _hbox_state: str = re.search(_box_pattern, haction.name).group()

                if _box_state == _hbox_state:
                    continue
            
            # We do not allow human to block the destination loc when robot action is release
            if 'release' in robot_action_name:
                _dloc: str = re.search(_loc_pattern, robot_action_name).group()
                _hloc: str = re.findall(_loc_pattern, haction.name)[1]

                if _dloc == _hloc:
                    continue
                
            # check if the destination loc is free or not
            if _intersect:
                d_loc_available: bool = self._check_exist_human_constraint(boxes=boxes,
                                                                           curr_state_lbl=curr_state_tuple,
                                                                           human_action_name=haction.name)

            if _intersect and d_loc_available:
                valid_move: bool = self.check_support_constraint(boxes=boxes,
                                                                 curr_state_lbl=curr_state_tuple,
                                                                 human_action_name=haction.name,
                                                                 robot_action_name=robot_action_name)
                if not valid_move:
                    continue
                
                # get add and del tuples 
                add_tuple = self.get_tuple_from_state(haction.add_effects)
                del_tuple = self.get_tuple_from_state(haction.del_effects)

                # construct the tuple for next state as per the human action
                hnext_tuple = list(set(curr_state_tuple) - set(del_tuple))
                hnext_tuple = tuple(sorted(list(set(hnext_tuple + list(add_tuple)))))

                # look up its corresponding formula
                next_sym_state: BDD = self.get_sym_state_from_tuple(hnext_tuple)

                if verbose:
                    self.print_human_edge(curr_state_tuple=curr_state_tuple,
                                          hnext_tuple=hnext_tuple,
                                          robot_action_name=robot_action_name,
                                          haction=haction,
                                          **kwargs)

                # add The edge to its corresponding action
                self.add_edge_to_action_tr(robot_action_name=robot_action_name,
                                           curr_state_tuple=curr_state_tuple,
                                           next_state_tuple=hnext_tuple,
                                           human_action_name=haction.name,
                                           **kwargs)

                # update valid ahuman actions list
                _hact_list.append(haction.name)

                # store the image in the next bucket
                if kwargs.get('curr_hint') is not None:
                    # add state & remaning human intervention to the next bucket
                    curr_hint: int = kwargs['curr_hint']
                    open_list[layer + 1] |= next_sym_state & self.predicate_sym_map_hint[curr_hint - 1]
                else:
                    # only add state to the next bucket
                    open_list[layer + 1] |= next_sym_state

                # get their corresponding lbls 
                # next_tuple_lbl = self.get_conds_from_state(state_tuple=hnext_tuple, only_world_conf=True)
                # next_lbl_sym = self.get_sym_state_lbl_from_tuple(next_tuple_lbl)
                self.sym_state_labels |= next_sym_state #& next_lbl_sym
        
        return _hact_list
    

    def create_transition_system_franka(self,
                                        boxes: List[str],
                                        state_lbls: list,
                                        add_exist_constr: bool = True,
                                        verbose: bool = False,
                                        plot: bool = False,
                                        **kwargs):
        """
         This method overrides the parent method. For every robot action, we loop over all the human actions.
        """
        
        if verbose:
            print(f"Creating TR for Actions:", *self.tr_action_idx_map.keys())

        self._create_sym_state_label_map(domain_lbls=state_lbls)

        open_list = defaultdict(lambda: self.manager.bddZero())

        closed = self.manager.bddZero()

        init_state_sym = self.sym_init_states

        # get the state lbls and create state and state lbl mappinng
        state_lbl = self.get_conds_from_state(state_tuple=self.predicate_sym_map_curr.inv[init_state_sym], only_world_conf=True)
        init_lbl_sym = self.get_sym_state_lbl_from_tuple(state_lbl)
        
        self.sym_state_labels |= init_state_sym & init_lbl_sym
        
        layer = 0

        # creating monolithinc tr bdd to keep track all the transition we are creating to detect nondeterminism
        if 'debug' in kwargs:
            self.mono_tr_bdd = self.manager.bddZero() 

        # no need to check if other boxes are placed at the destination loc during transfer and release as there is only one object
        if len(boxes) == 1:
            add_exist_constr = False

        open_list[layer] |= init_state_sym

        while not open_list[layer].isZero():
            # remove all states that have been explored
            open_list[layer] = open_list[layer] & ~closed

            # If unexpanded states exist ...
            if not open_list[layer].isZero():
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer]

                if verbose:
                    print(f"******************************* Layer: {layer}*******************************")

                # get all the states
                sym_state = self._convert_state_lbl_cube_to_func(dd_func= open_list[layer], prod_curr_list=self.sym_vars_curr)
                for state in sym_state:
                    curr_state_tuple = self.predicate_sym_map_curr.inv[state]
                    
                    _valid_pre_list = []
                    # compute the image of the TS states
                    for action in self.task.operators:
                        # we skip human moves as we manually loop over afterwards 
                        if 'human' in action.name:
                            continue

                        # set action feasbility flag to True - used during transfer and release action to check the des loc is empty
                        action_feas: bool = True
                        pre_tuple = self.get_tuple_from_state(action.preconditions)
                        _necc_robot_conf = self.get_conds_from_state(pre_tuple, only_robot_conf=True)

                        _intersect: bool = set(pre_tuple).issubset(curr_state_tuple)

                        if _intersect:
                            # get valid pres from current state tuple
                            pre_robot_conf = self.get_conds_from_state(curr_state_tuple, only_robot_conf=True)
                            pre_robot_conf = tuple(set(pre_robot_conf).intersection(_necc_robot_conf))
                            pre_world_conf = self.get_conds_from_state(curr_state_tuple, only_world_conf=True)

                            _valid_pre = sorted(pre_robot_conf + pre_world_conf)
                            
                            if tuple(_valid_pre) != curr_state_tuple:
                                _valid_pre_sym = self.predicate_sym_map_curr[tuple(_valid_pre)]
                                # check if this state has already being explored or not
                                if not (_valid_pre_sym & closed).isZero():
                                    continue
                                _valid_pre_list.append(_valid_pre_sym)

                            # add existential constraints to transfer and relase action
                            if add_exist_constr and (('transfer' in action.name) or ('release' in action.name)):
                                action_feas = self._check_exist_constraint(boxes=boxes,
                                                                           curr_state_lbl=_valid_pre,
                                                                           action_name=action.name)
                            
                            if not action_feas:
                                continue

                            # get add and del tuples 
                            add_tuple = self.get_tuple_from_state(action.add_effects)
                            del_tuple = self.get_tuple_from_state(action.del_effects)

                            # construct the tuple for next state
                            next_tuple = list(set(_valid_pre) - set(del_tuple))
                            next_tuple = tuple(sorted(list(set(next_tuple + list(add_tuple)))))

                            # look up its corresponding formula
                            next_sym_state: BDD = self.predicate_sym_map_nxt[next_tuple]

                            if verbose:
                                cstate = self.get_state_from_tuple(state_tuple=tuple(_valid_pre))
                                nstate = self.get_state_from_tuple(state_tuple=next_tuple)
                                print(f"Adding Robot edge: {cstate} -------{action.name}------> {nstate}")
                            
                            # add human moves, if any. UNder human action we evolve as per the robot action and human action.
                            env_edge_acts: list =  self.add_human_moves(robot_action_name=action.name,
                                                                        open_list=open_list,
                                                                        curr_state_tuple=tuple(_valid_pre),
                                                                        robot_nxt_tuple=next_tuple,
                                                                        layer=layer,
                                                                        boxes=boxes,
                                                                        verbose=verbose,
                                                                        **kwargs)
                            
                            # add The edge to its corresponding action
                            self.add_edge_to_action_tr(robot_action_name=action.name,
                                                       curr_state_tuple=tuple(_valid_pre),
                                                       next_state_tuple=next_tuple,
                                                       valid_hact_list=env_edge_acts,
                                                       **kwargs)

                            # get their corresponding lbls 
                            next_tuple_lbl = self.get_conds_from_state(state_tuple=next_tuple, only_world_conf=True)
                            next_lbl_sym = self.get_sym_state_lbl_from_tuple(next_tuple_lbl)
                            self.sym_state_labels |= next_sym_state & next_lbl_sym

                            # store the image in the next bucket
                            open_list[layer + 1] |= next_sym_state

                    for _val_pre_sym in _valid_pre_list:
                        # add them the observation bdd
                        _valid_pre_lbl = self.get_conds_from_state(state_tuple=self.predicate_sym_map_curr.inv[_val_pre_sym],
                                                                   only_world_conf=True)
                        _valid_pre_lbl_sym = self.get_sym_state_lbl_from_tuple(_valid_pre_lbl)
                        self.sym_state_labels |= _val_pre_sym & _valid_pre_lbl_sym

                        closed |= _val_pre_sym
                
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
                 dfa_state_vars: List[BDD]):
        self.sym_vars_hint: List[BDD] = human_int_vars
        self.state_lbls = ts_state_lbls
        super().__init__(curr_vars, lbl_vars, robot_action_vars, human_action_vars, task, domain, ts_state_map, ts_states, manager)
        
        self.max_hint: int = max_human_int

        self.predicate_sym_map_hint: bidict = {}

        self._initialize_bdd_for_human_int()

        # store the bdd associated with each state vars in this list. The index corresonds to its number
        
        self.tr_state_bdds = [self.manager.bddZero() for _ in range(sum([len(listElem) for listElem in self.sym_vars_lbl]) + len(self.sym_vars_curr) + len(self.sym_vars_hint))]

        # create adj map. Useful when rolling out strategy with human intervention for sanity checking
        self.adj_map = defaultdict(lambda: defaultdict(lambda : {'h': [], 'r': []}))

        # index to determine where the state vars start 
        self.state_start_idx: int =  len(self.sym_vars_human) + len(self.sym_vars_robot) + len(dfa_state_vars)

        self.hint_cube = reduce(lambda x, y: x & y, self.sym_vars_hint)
        self.state_cube = reduce(lambda x, y: x & y, self.sym_vars_curr)
        self.lbl_cube = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list])

    

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
                              robot_action_name: str,
                              curr_state_tuple: tuple,
                              next_state_tuple: tuple,
                              human_action_name: str = '',
                              valid_hact_list: List[str] = None,
                              **kwargs) -> None:
        if valid_hact_list is None:
            valid_hact_list = []
        elif isinstance(valid_hact_list, list):
            # if there are any valid human edges from curr state
            if len(valid_hact_list) > 0:
                valid_hact: List[BDD] = [self.predicate_sym_map_human[ha] for ha in valid_hact_list]
                no_human_move: BDD = ~(reduce(lambda x, y: x | y, valid_hact))
            else:
                no_human_move: BDD = self.manager.bddOne()

        else:
            warnings.warn("Invalid Default args type when constructing the Symbolic Franka Abstraction. FIX THIS!!!")
            sys.exit(-1)

        curr_hint: int = kwargs['curr_hint']
        curr_state_sym: BDD = self.get_sym_state_from_tuple(curr_state_tuple)
        nxt_state_sym: BDD = self.get_sym_state_from_tuple(next_state_tuple)

        if human_action_name != '':
            nxt_state_sym = nxt_state_sym & self.predicate_sym_map_hint[curr_hint - 1]
            _tr_idx: int = self.tr_action_idx_map.get(human_action_name)
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & \
                            self.predicate_sym_map_human[human_action_name] & self.predicate_sym_map_hint[curr_hint]).isZero()
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {self.get_state_from_tuple(curr_state_tuple)}[{curr_hint}] ---{human_action_name}---> {self.get_state_from_tuple(next_state_tuple)}[{curr_hint - 1}]")
                    self.mono_tr_bdd |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & \
                                self.predicate_sym_map_human[human_action_name] & self.predicate_sym_map_hint[curr_hint]
        else:
            nxt_state_sym = nxt_state_sym & self.predicate_sym_map_hint[curr_hint]
            _tr_idx: int = self.tr_action_idx_map.get(robot_action_name)
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd &  curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & \
                            no_human_move & self.predicate_sym_map_hint[curr_hint]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Robot Action: {self.get_state_from_tuple(curr_state_tuple)}[{curr_hint}] ---{robot_action_name}---> {self.get_state_from_tuple(next_state_tuple)}[{curr_hint}]")
                
                    self.mono_tr_bdd |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & \
                                no_human_move & self.predicate_sym_map_hint[curr_hint]             


        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(nxt_state_sym.cube()):
            if var == 1 and self.manager.bddVar(_idx) in kwargs['prod_curr_list']:# [*self.sym_vars_curr, *self.sym_vars_hint]:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                # if human intervenes then the edge looks like (robot-action) & (human move b# l# l#)
                if human_action_name != '':
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & \
                         self.predicate_sym_map_human[human_action_name] & self.predicate_sym_map_hint[curr_hint]
                # if human does not intervene then the edge looks like (robot-action) & not(valid human moves)
                else:
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & self.predicate_sym_map_robot[robot_action_name] & \
                         no_human_move & self.predicate_sym_map_hint[curr_hint]

            
            elif var == 2 and self.manager.bddVar(_idx) in self.sym_vars_curr:
                warnings.warn("Encountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)
        
        if human_action_name != '':
            self.adj_map[curr_state_tuple][curr_hint]['h'].append(next_state_tuple)
        else:
            self.adj_map[curr_state_tuple][curr_hint]['r'].append(next_state_tuple)
        
        # update edge count 
        self.ecount += 1
    

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


    
    def print_human_edge(self, curr_state_tuple: tuple, hnext_tuple: tuple, robot_action_name: str, haction, **kwargs):
        """
         Override base printing method to include remaining human intervention associated with each state.
        """
        
        cstate = self.get_state_from_tuple(state_tuple=tuple(curr_state_tuple))
        nstate = self.get_state_from_tuple(state_tuple=hnext_tuple)
        print(f"Adding Human edge: {cstate}[{kwargs['curr_hint']}] -------{haction.name}------> {nstate}[{kwargs['curr_hint'] - 1}]")


    def create_transition_system_franka(self,
                                        boxes: List[str],
                                        state_lbls: list,
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
        if verbose:
            print(f"Creating TR for Actions:", *self.tr_action_idx_map.keys())

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

        while not open_list[layer].isZero():
            # remove all states that have been explored
            open_list[layer] = open_list[layer] & ~closed

            # If unexpanded states exist ...
            if not open_list[layer].isZero():
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer]

                if verbose:
                    print(f"******************************* Layer: {layer}*******************************")

                # get all the states and their corresponding remaining human intervention 
                sym_state = self._convert_state_lbl_cube_to_func(dd_func= open_list[layer], prod_curr_list=prod_curr_list)
                for state in sym_state:
                    curr_state_tuple, curr_hint = self.get_state_tuple_from_sym_state(sym_state=state, sym_lbl_xcube_list=sym_lbl_xcube_list)

                    # compute the image of the TS states
                    for action in self.task.operators:
                        # we skip human moves as we manually loop over afterwards 
                        if 'human' in action.name:
                            continue

                        # set action feasbility flag to True - used during transfer and release action to check the des loc is empty
                        action_feas: bool = True
                        pre_tuple = self.get_tuple_from_state(action.preconditions)
                        _necc_robot_conf = self.get_conds_from_state(pre_tuple, only_robot_conf=True)

                        _intersect: bool = set(pre_tuple).issubset(curr_state_tuple)

                        if _intersect:
                            # get valid pres from current state tuple
                            pre_robot_conf = self.get_conds_from_state(curr_state_tuple, only_robot_conf=True)
                            pre_robot_conf = tuple(set(pre_robot_conf).intersection(_necc_robot_conf))
                            pre_world_conf = self.get_conds_from_state(curr_state_tuple, only_world_conf=True)

                            _valid_pre: list = sorted(pre_robot_conf + pre_world_conf)

                            # add existential constraints to transfer and relase action
                            if add_exist_constr and (('transfer' in action.name) or ('release' in action.name)):
                                action_feas = self._check_exist_constraint(boxes=boxes,
                                                                           curr_state_lbl=_valid_pre,
                                                                           action_name=action.name)
                            
                            if not action_feas:
                                continue

                            # get add and del tuples 
                            add_tuple = self.get_tuple_from_state(action.add_effects)
                            del_tuple = self.get_tuple_from_state(action.del_effects)

                            # construct the tuple for next state
                            next_tuple = list(set(_valid_pre) - set(del_tuple))
                            next_tuple = tuple(sorted(list(set(next_tuple + list(add_tuple)))))

                            # look up its corresponding formula
                            next_sym_state: BDD = self.get_sym_state_from_tuple(next_tuple)

                            if verbose:
                                cstate: tuple = self.get_state_from_tuple(state_tuple=tuple(_valid_pre))
                                nstate: tuple = self.get_state_from_tuple(state_tuple=next_tuple)
                                print(f"Adding Robot edge: {cstate}[{curr_hint}] -------{action.name}------> {nstate}[{curr_hint}]")
                            
                            env_edge_acts = []

                            # add human moves, if any. Under human action we evolve as per the robot action and human action.
                            if curr_hint > 0:
                                env_edge_acts: list =  self.add_human_moves(robot_action_name=action.name,
                                                                            open_list=open_list,
                                                                            curr_state_tuple=tuple(_valid_pre),
                                                                            robot_nxt_tuple=next_tuple,
                                                                            layer=layer,
                                                                            boxes=boxes,
                                                                            verbose=verbose,
                                                                            curr_hint=curr_hint,
                                                                            prod_curr_list=prod_curr_list,
                                                                            **kwargs)
                            
                            # add The edge to its corresponding action
                            self.add_edge_to_action_tr(robot_action_name=action.name,
                                                       curr_state_tuple=tuple(_valid_pre),
                                                       next_state_tuple=next_tuple,
                                                       curr_hint=curr_hint,
                                                       valid_hact_list=env_edge_acts,
                                                       prod_curr_list=prod_curr_list,
                                                       **kwargs)

                            # get their corresponding lbls 
                            self.sym_state_labels |= next_sym_state 

                            # store the image in the next bucket
                            open_list[layer + 1] |= next_sym_state & self.predicate_sym_map_hint[curr_hint]
                
                layer += 1
        
        if kwargs['print_tr'] is True:
            self._print_plot_tr(plot=plot)