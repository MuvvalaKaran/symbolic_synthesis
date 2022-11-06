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

        # for safety remove variable deprecated vars from parent class
        del self.sym_vars_action
        del self.predicate_sym_map_act


    
    def _initialize_bdds_for_actions(self):
        """
         This function intializes the mapping for both the robot and human actions to their corresponding
          Boolean formulas expressed using robot vars (o / output vairables/controllable vars) and
          human vars (i / input vairables/uncontrollable vars), respectively. 
        """
        for _idx, var in enumerate([self.sym_vars_human, self.sym_vars_robot]):
        
            # create all combinations of 1-true and 0-false
            boolean_str = list(product([1, 0], repeat=len(var)))
            _node_int_map =  {}

            iter_count: int = 0
            for state in self.actions:
                if _idx == 0:
                    _node_int_map = {state: boolean_str[index] for index, state in enumerate(['(human-move)', '(no-human-move)'])}
                    break
                elif _idx == 1 and 'human' not in state:
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
            
            if _idx == 0:
                self.predicate_sym_map_human = bidict(_node_int_map)
            else:
                self.predicate_sym_map_robot = bidict(_node_int_map)
        
    

    def add_edge_to_action_tr(self, robot_action_name: str, curr_state_tuple: tuple, next_state_tuple: tuple, human_intervene: bool = False) -> None:
        """
         While creating the TR for the two-player game, we add both human and robot action. We do encode every parameterized human move, i.e.,
          human-move b0 l6 l7 is simply added as human-move. This, restricts the no. of binary variables required to encode humans to two, i.e., 
          human intervene (human-move) and human not intervene (no-human-move)
        """
        curr_state_sym: BDD = self.predicate_sym_map_curr[curr_state_tuple]
        nxt_state_sym: BDD = self.predicate_sym_map_curr[next_state_tuple]

        # get the corresponding symbolic actions
        sym_action_robot: BDD = self.predicate_sym_map_robot[robot_action_name]
        sym_action_human: BDD = self.predicate_sym_map_human['(human-move)'] if human_intervene else self.predicate_sym_map_human['(no-human-move)']

        # for every boolean var in nxt_state check if it high or low. If high add it curr state and the correpsonding action to its BDD
        for _idx, var in enumerate(nxt_state_sym.cube()):
            if var == 1 and self.manager.bddVar(_idx) in self.sym_vars_curr:
                _state_idx: int = _idx - self.state_start_idx
                assert _state_idx >= 0, "Error constructing the Partitioned Transition Relation."
                
                self.tr_state_bdds[_state_idx] |= curr_state_sym & sym_action_robot & sym_action_human
            
            elif var == 2 and self.manager.bddVar(_idx) in self.sym_vars_curr:
                warnings.warn("Ecvountered an ambiguous varible during TR construction. FIX THIS!!!")
                sys.exit(-1)
    

    def add_human_moves(self, robot_action_name: str, open_list: dict, curr_state_tuple: tuple, layer: int, verbose: bool = False):
        """
         A function that loops over all the human moves, check if the preconditions are met, if yes then compute the next state and add it to the TR.  
        """
        
        for haction in self.task.operators:
            if 'human' not in haction.name:
                continue
            
            # check is the preconditions of human action are satisfied or not
            _pre_tuple = self.get_tuple_from_state(haction.preconditions)  # cardinality of human move's precondition is always 1
            _intersect: bool = set(_pre_tuple).issubset(curr_state_tuple)

            # we do not allow the human move the obj the robot is currently grasping
            if 'grasp' in robot_action_name:
                _box_pattern = "[b|B][\d]+"
                _box_state: str = re.search(_box_pattern, robot_action_name).group()
                _hbox_state: str = re.search(_box_pattern, haction.name).group()

                if _box_state == _hbox_state:
                    continue
            
            # We do not allow human to block the destination loc when robot action is release
            if 'release' in robot_action_name:
                _loc_pattern = "[l|L][\d]+"
                _dloc: str = re.search(_loc_pattern, robot_action_name).group()
                _hloc: str = re.findall(_loc_pattern, haction.name)[1]

                if _dloc == _hloc:
                    continue

            if _intersect:
                # get add and del tuples 
                add_tuple = self.get_tuple_from_state(haction.add_effects)
                del_tuple = self.get_tuple_from_state(haction.del_effects)

                # construct the tuple for next state
                next_tuple = list(set(curr_state_tuple) - set(del_tuple))
                next_tuple = tuple(sorted(list(set(next_tuple + list(add_tuple)))))

                # look up its corresponding formula
                next_sym_state: BDD = self.predicate_sym_map_nxt[next_tuple]

                if verbose:
                    cstate = self.get_state_from_tuple(state_tuple=tuple(curr_state_tuple))
                    nstate = self.get_state_from_tuple(state_tuple=next_tuple)
                    print(f"Adding Human edge: {cstate} -------{haction.name}------> {nstate}")

                # add The edge to its corresponding action
                self.add_edge_to_action_tr(robot_action_name=robot_action_name,
                                           curr_state_tuple=curr_state_tuple,
                                           next_state_tuple=next_tuple,
                                           human_intervene=True)

                # store the image in the next bucket
                open_list[layer + 1] |= next_sym_state

    

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

                            # add human moves, if any. . .
                            self.add_human_moves(robot_action_name=action.name,
                                                open_list=open_list,
                                                curr_state_tuple=tuple(_valid_pre),
                                                layer=layer,
                                                verbose=verbose)

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
                            
                            # add The edge to its corresponding action
                            self.add_edge_to_action_tr(robot_action_name=action.name,
                                                       curr_state_tuple=tuple(_valid_pre),
                                                       next_state_tuple=next_tuple,
                                                       human_intervene=False)


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