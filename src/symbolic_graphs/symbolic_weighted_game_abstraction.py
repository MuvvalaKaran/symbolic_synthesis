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


class DynWeightedPartitionedFrankaAbs():
    """
     A class that constructs symbolic Two-player Transition Relation walogn with weights.
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
                 weight_dict: dict,
                 seg_actions: dict,
                 ts_state_lbls: list,
                 dfa_state_vars: List[ADD],
                 sup_locs: List[str],
                 top_locs: List[str],
                 **kwargs):
        self.sym_vars_curr: List[ADD] = curr_vars
        self.sym_vars_lbl: List[ADD] = lbl_vars
        self.sym_vars_human: List[ADD] = human_action_vars
        self.sym_vars_robot: List[ADD] = robot_action_vars

        self.init: frozenset = task.initial_state
        self.ts_states: dict = ts_states
        self.state_lbls = ts_state_lbls
        self.weight_dict: Dict[str, List[ADD]] = weight_dict

        self.pred_int_map: dict = ts_state_map

        self.task: dict = task
        self.domain: dict = domain
        self.manager = manager
        
        self.actions: Dict[str, list] = seg_actions
        self.tr_action_idx_map: dict = {}
        self.sym_init_states: ADD = manager.addZero()
        
        self.sym_state_labels: ADD = manager.addZero()
        self.sym_tr_actions: list = []
        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl: bidict = {}


        self._create_sym_var_map()
        self._initialize_adds_for_actions()
        self._initialize_sym_init_goal_states()

        # index to determine where the state vars start 
        self.state_start_idx: int =  len(self.sym_vars_human) + len(self.sym_vars_robot) + len(dfa_state_vars)

        # create adj map. Useful when rolling out strategy with human intervention for sanity checking
        self.adj_map = defaultdict(lambda: defaultdict(lambda: {}))

        # refered during graph of utility construction
        self.org_adj_map = defaultdict(lambda: defaultdict(lambda: {}))
        # edge counter
        self.ecount: int = 0

        self.state_cube = reduce(lambda x, y: x & y, self.sym_vars_curr)
        self.lbl_cube = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.sym_vars_lbl for lbl in sym_vars_list])
        self.sys_cube: ADD = reduce(lambda x, y: x & y, self.sym_vars_robot)
        self.env_cube: ADD = reduce(lambda x, y: x & y, self.sym_vars_human)

        # adding support and top location. Useful during arch construction to chekc for valid human intervention.
        self.sup_locs = sup_locs
        self.top_locs = top_locs


    def _initialize_sym_init_goal_states(self):
        """
         Initialize the inital states of the Transition system with their corresponding symbolic init state vairants.
        """
        self._create_sym_state_label_map(domain_lbls=self.state_lbls)
        init_tuple = self.get_tuple_from_state(self.init)
        
        self.sym_init_states = self.get_sym_state_from_tuple(init_tuple)
    

    def _create_sym_state_label_map(self, domain_lbls):
        """
         Loop through all the facts that are reachable and assign a boolean funtion to it.
        """
        # loop over each box and create its corresponding boolean formula 
        for b_id, preds in domain_lbls.items():
            # get its corresponding boolean vars
            _id: int = int(re.search("\d+", b_id).group())
            _tmp_vars_list = self.sym_vars_lbl[_id]

            # TODO: When the boxes are out of sequence, say only b0 abd b2 exists, this for loop fails. FIX THIS!!!
            # create all combinations of 1-true and 0-false
            boolean_str = list(product([1, 0], repeat=len(_tmp_vars_list)))

            _node_int_map_lbl = bidict({state: boolean_str[index] for index, state in enumerate(preds)})

            assert len(boolean_str) >= len(_node_int_map_lbl), "FIX THIS: Looks like there are more lbls that boolean variables!"

            # loop over all the boolean string and convert them to their respective add vars
            for _key, _value in _node_int_map_lbl.items():
                _lbl_val_list = []
                for _idx, _ele in enumerate(_value):
                    if _ele == 1:
                        _lbl_val_list.append(_tmp_vars_list[_idx])
                    else:
                        _lbl_val_list.append(~_tmp_vars_list[_idx])
                
                _bool_func_curr = reduce(lambda a, b: a & b, _lbl_val_list)

                # update bidict accordingly
                _node_int_map_lbl[_key] = _bool_func_curr
            
            self.predicate_sym_map_lbl.update(_node_int_map_lbl)
        
        self.predicate_sym_map_lbl = bidict(self.predicate_sym_map_lbl)


    def _create_sym_var_map(self):
        """
         Loop through all the facts that are reachable and assign a boolean function to it.
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_vars_curr)))

        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(self.ts_states)})

        assert len(boolean_str) >= len(_node_int_map_curr), "FIX THIS: Looks like there are more Facts than boolean variables!"

        # loop over all the boolean strings and convert them respective add vars
        for _key, _value in _node_int_map_curr.items():
            _curr_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _curr_val_list.append(self.sym_vars_curr[_idx])
                else:
                    _curr_val_list.append(~self.sym_vars_curr[_idx])
            
            _bool_func_curr = reduce(lambda a, b: a & b, _curr_val_list)

            # update bidict accordingly
            _node_int_map_curr[_key] = _bool_func_curr
        
        self.predicate_sym_map_curr = bidict(_node_int_map_curr)
    

    def _initialize_adds_for_actions(self):
        """
         This function intializes the mapping for both the robot and human actions to their corresponding
          Boolean formulas expressed using robot vars (o / output vairables/controllable vars) and
          human vars (i / input vairables/uncontrollable vars), respectively. 
        """
        # for _pidx, var in enumerate([self.sym_vars_human, self.sym_vars_robot]):
        for key, acts in self.actions.items():
            if key == 'human':
                sym_vars = self.sym_vars_human
            else:
                sym_vars = self.sym_vars_robot

            # create all combinations of 1-true and 0-false
            boolean_str = list(product([1, 0], repeat=len(sym_vars)))
            _node_int_map =  {action: boolean_str[index] for index, action in enumerate(acts)}

            assert len(boolean_str) >= len(_node_int_map), "FIX THIS: Looks like there are more Actions than boolean variables!"

            # loop over all the boolean strings and convert them respective add vars
            for _key, _value in _node_int_map.items():
                _act_val_list = []
                for _idx, _ele in enumerate(_value):
                    if _ele == 1:
                        _act_val_list.append(sym_vars[_idx])
                    else:
                        _act_val_list.append(~sym_vars[_idx])
                    
                    _bool_func_curr = reduce(lambda a, b: a & b, _act_val_list)

                    # update bidict accordingly
                    _node_int_map[_key] = _bool_func_curr
            
            if key == 'human':
                self.predicate_sym_map_human = bidict(_node_int_map)
            else:
                self.predicate_sym_map_robot = bidict(_node_int_map)
        
        self.initialize_sym_tr_action_list()
    
    
    def initialize_sym_tr_action_list(self):
        """
         Create a list of Zero ADDs for each paramterized action (Robot and Human) defined in the pddl domain file.

         NOTE: One key difference in the sym TR in the Quantitative abstraction and the Qualitative abstarction is that
          we do not create TR for human actions in this Class. This is avoid redundance caused nby Human actions. The same should be updated in the 
          Unbounded Human abstraction. 
        """
        # initiate ADDs for all the action 
        action_idx_map = bidict()
        num_of_acts: int = 0

        for act in self.task.operators:
            if 'human' not in act.name:
                action_idx_map[act.name] = num_of_acts
                num_of_acts += 1
        
        self.tr_action_idx_map = action_idx_map
        
        num_ts_state_vars: int = sum([len(listElem) for listElem in self.sym_vars_lbl]) + len(self.sym_vars_curr)
        self.sym_tr_actions = [[self.manager.addZero() for _ in range(num_ts_state_vars)] for _ in range(num_of_acts)]

    
    def get_conds_from_state(self, state_tuple: tuple, only_world_conf: bool = False, only_robot_conf: bool = False) -> Tuple[int]:
        """
         A function that loops through the state tuple, and returns the tuple corresponding the robot conf or world ocnf.

         @param: only_world_conf - Set this to true if you only want to return `on` predicates (box locations)
         @param: only_robot_conf - Set this to true if you want to return predicates related to robot conf. (ready, to-obj, holding, to-loc) 
        """
        preds = self.get_state_from_tuple(state_tuple=state_tuple)

        _int_tuple = []
        for pred in preds:
            if only_world_conf:
                if 'on' in pred:
                    _int_tuple.append(self.pred_int_map[pred])

            elif only_robot_conf:
                if 'on' not in pred:
                    _int_tuple.append(self.pred_int_map[pred])
        
        return tuple(sorted(_int_tuple))
    

    def get_sym_state_from_tuple(self, state_lbl_tuple: tuple) -> ADD:
        """
         A function that converts the corresponding state lbl tuple to its explicit predicate form,
          looks up its corresponding boolean formula, and return the conjunction of all the boolean formula
        """
        # get the explicit preds
        exp_lbls = self.get_state_from_tuple(state_tuple=state_lbl_tuple)

        _sym_lbls_list = []
        for lbl in exp_lbls:
            if lbl in self.predicate_sym_map_lbl:
                _sym_lbls_list.append(self.predicate_sym_map_lbl[lbl])
            else:
                _sym_lbls_list.append(self.predicate_sym_map_curr[lbl])
        
        sym_lbl = reduce(lambda x, y: x & y, _sym_lbls_list)

        assert not sym_lbl.isZero(), "Error constructing the symbolic lbl associated with each state. FIX THIS!!!"

        return sym_lbl
    

    def get_sym_state_from_exp_states(self, exp_states: List[BDD]) -> ADD:
        """
         A function that iterates over the list of explicit states, looks up its corresponding boolean formula,
          and returns the conjunction of all the boolean formula
        """
        _sym_lbls_list = []
        for lbl in exp_states:
            if lbl in self.predicate_sym_map_lbl:
                _sym_lbls_list.append(self.predicate_sym_map_lbl[lbl])
            else:
                _sym_lbls_list.append(self.predicate_sym_map_curr[lbl])
        
        sym_lbl = reduce(lambda x, y: x & y, _sym_lbls_list)

        assert not sym_lbl.isZero(), "Error constructing the symbolic lbl associated with each state. FIX THIS!!!"
        assert list(sym_lbl.generate_cubes())[0][1] == 1, "Error constructing Sym state as 0-1 ADD. FIX THIS!!!"

        return sym_lbl


    def get_sym_state_lbl_from_tuple(self, state_lbl_tuple: tuple) -> ADD:
        """
         A function that converts the corresponding state lbl tuple to its explicit predicate form,
          looks up its corresponding boolean formula, and return the conjunction of all the boolean formula
        """
        # get the explicit preds
        exp_lbls = self.get_state_from_tuple(state_tuple=state_lbl_tuple)

        _sym_lbls_list = [self.predicate_sym_map_lbl[lbl] for lbl in exp_lbls]
        
        sym_lbl = reduce(lambda x, y: x & y, _sym_lbls_list)

        assert not sym_lbl.isZero(), "Error constructing the symbolic lbl associated with each state. FIX THIS!!!"

        return sym_lbl


    def get_tuple_from_state(self, preds: list) -> Tuple[int]:
        """
         Given, a predicate tuple, this function return the corresponding state tuple
        """
        _int_tuple = [self.pred_int_map[pred] for pred in preds]

        return tuple(sorted(_int_tuple))
    

    def get_state_from_tuple(self, state_tuple: tuple) -> List[str]:
        """
         Given a predicate tuple, this function returns the corresponding state tuple
        """
        if isinstance(state_tuple, tuple):
            _states = [self.pred_int_map.inv[state] for state in state_tuple]
        else:
            _states = self.pred_int_map.inv[state_tuple[0]]

        return _states
    

    def get_state_tuple_from_sym_state(self, sym_state: ADD, sym_lbl_xcube_list: List[ADD]) -> tuple:
        """
         A function that loops over the entire sym state, extracts the corresponding predicates, looks up their index and returns the tuple
        """
        curr_state_name = self.predicate_sym_map_curr.inv[sym_state.existAbstract(self.lbl_cube)]
        curr_state_int = self.pred_int_map.get(curr_state_name)
        
        _lbl_list = []
        
        for idx in range(len(self.sym_vars_lbl)):
            # create a cube of the rest of the dfa vars
            exist_dfa_cube = self.manager.addOne()
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


    def check_support_constraint(self,
                                  boxes: List[str],
                                  curr_state_lbl: tuple,
                                  human_action_name: str,
                                  robot_action_name: str, **kwargs) -> bool:
        """
         Given the current human move check if the box being manipulated by the human is at a support location or not.
          If yes, check if there is something in "top loc" or not.
        """
        _box_pattern = "[b|B][\d]+"
        _loc_pattern = "[l|L][\d]+"
        _chloc: List[str] = re.findall(_loc_pattern, human_action_name)[0]
        _box_state: str = re.search(_box_pattern, human_action_name).group()
        # check if the box is in support loc and has another box in top location, i.e,
        # if box1 is being manipulated, get the list of rest of boxes (that have to be grounded) at this instance

        if _chloc in self.sup_locs:  # if the human move is from current loc
            tmp_copy = copy.deepcopy(boxes)
            tmp_copy.remove(_box_state)

            # create predicates that say on b0 l1 (l1 is the destination in the action)
            on_preds = [self.pred_int_map[f'(on {bid} {tloc})'] for bid in tmp_copy for tloc in self.top_locs]

            # check if a box exists on "top" in the curr state lbl or after the completion of the robot action
            if set(on_preds).intersection(set(curr_state_lbl)):
                return False
            
            # for Unbounded Abstraction
            if 'release' in robot_action_name:
                _dloc: str = re.search(_loc_pattern, robot_action_name).group()
                if _dloc in self.top_locs:
                    return False

        return True


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
        if dloc in self.top_locs:
            return False

        # if box1 is being manipulated, get the list of rest of boxes (that have to be grounded) at this instance
        # create predicates that say on b0 l1 (l1 is the destination in the action)
        on_preds = [self.pred_int_map[f'(on {bid} {dloc})'] for bid in boxes if bid != _box_state]
        
        if set(on_preds).intersection(set(curr_state_lbl)):
            return False

        return True


    def _check_exist_constraint(self, boxes: List[str], curr_state_lbl: tuple, action_name: str) -> bool:
        """
        A helper function that take as input the state label (on b0 l0)(on b1 l1) and the action name,
         extracts the destination location from action name and its corresponding world conf tuple.
         We then take the intersection of the  curr_state_lbl & corresponding world conf tuple.
         
        Return False if intersection is non-empty else True
        """
        _loc_pattern = "[l|L][\d]+"
        _box_pattern = "[b|B][\d]+"

        if 'transfer' in action_name: 
            locs: List[str] = re.findall(_loc_pattern, action_name)
            if 'else' in action_name:
                return f'transfer {locs[0]}'
            else:
                return f'transfer {locs[1]}'
        elif 'release' in action_name:
            box_state: str = re.search(_box_pattern, action_name).group()
            dloc: str = re.search(_loc_pattern, action_name).group()
            
        tmp_copy = [_b for _b in boxes if _b != box_state]

        # create predicates that say on b0 l1 (l1 is the destination in the action)
        on_preds = [self.pred_int_map[f'(on {bid} {dloc})'] for bid in tmp_copy]
        
        if set(on_preds).intersection(set(curr_state_lbl)):
            return False

        return True
        

    def create_human_edges(self,
                            human_actions,
                            curr_sym_state,
                            curr_state_tuple,
                            mod_act_dict: Dict,
                            sym_lbl_cubes,
                            open_list,
                            layer,
                            boxes,
                            prod_curr_list,
                            verbose,
                            robot_action_name,
                            next_exp_state,
                            **kwargs):
        """
         A function that loops through all the human edges and adds the valid moves to the symbolic TR.
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
                                                  next_exp_state=next_exp_state,
                                                  robot_action_name=robot_action_name,
                                                  mod_act_dict=mod_act_dict,
                                                  layer=layer,
                                                  boxes=boxes,
                                                  verbose=verbose,
                                                  prod_curr_list=prod_curr_list,
                                                  **kwargs)
            if valid_act:
                env_edge_acts.append(action.name)

        return env_edge_acts
    

    def print_human_edge(self, curr_exp_states: List[str], hnext_exp_states: List[str], haction_name, robot_action_name):
        """
         A helper function solely for printing an human intervention edge.
        """
        print(f"Adding Human edge: {curr_exp_states} -------{robot_action_name} & {haction_name}------> {hnext_exp_states}")
    

    def add_human_moves(self,
                        haction,
                        open_list: dict,
                        curr_state_tuple: tuple,
                        curr_exp_states: List[str],
                        next_exp_state: List[str],
                        curr_state_sym: BDD,
                        mod_act_dict,
                        curr_rob_conf: str,
                        robot_action_name: str,
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
        
        elif 'release' in robot_action_name:
            _dloc: str = re.search(_loc_pattern, robot_action_name).group()
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
                                                             robot_action_name=robot_action_name,
                                                             **kwargs)
            if not valid_move:
                return False

            next_exp_state = list(haction.apply(state=frozenset(next_exp_state)))

            if 'transit' in robot_action_name:
                _hcloc: str = re.findall(_loc_pattern, haction.name)[0]
                
                 # for actions of type (transit b# else l#) or (tansit b# l# else) 
                if 'else' in robot_action_name:
                    _dloc: str = robot_action_name.split(' ')[-1][:-1]
                else:
                    # check if you are trnasiting from 'else' to l# or from l# to 'else'
                    _dloc: str = re.findall(_loc_pattern, robot_action_name)[1]

                _box_state: str = re.search(_box_pattern, robot_action_name).group()

                if _hcloc == _dloc:
                    # remove to-obj prediacte
                    next_exp_state = list(set(next_exp_state) - set([f'(to-obj {_box_state})']))
                    next_exp_state = next_exp_state + [f'(ready {_dloc})']
            
            next_sym_state: ADD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)


            if verbose:
                self.print_human_edge(curr_exp_states=list(curr_exp_states),
                                      hnext_exp_states=next_exp_state,
                                      haction_name=haction.name,
                                      robot_action_name=robot_action_name)

            # add The edge to its corresponding action
            self.add_edge_to_action_tr(curr_state_tuple=curr_state_tuple,
                                       curr_state_sym=curr_state_sym,
                                       next_state_tuple=next_exp_state,
                                       mod_act_dict=mod_act_dict,
                                       nxt_state_sym=next_sym_state,
                                       human_action_name=haction.name,
                                       robot_action_name=robot_action_name,
                                       **kwargs)

            # store the image in the next bucket
            open_list[layer + 1] |= next_sym_state

            # get their corresponding lbls 
            self.sym_state_labels |= next_sym_state 
    
            return True
        
        return False
    

    def add_edge_to_action_tr(self,
                              curr_state_tuple: tuple,
                              next_state_tuple: tuple,
                              curr_state_sym: ADD,
                              nxt_state_sym: ADD,
                              mod_act_dict: dict,
                              curr_str_state: List[str]='',
                              next_str_state: List[str]='',
                              robot_action_name: str = '',
                              human_action_name: str = '',
                              valid_hact_list: ADD = None,
                              **kwargs):
        if valid_hact_list is not None:
            assert isinstance(valid_hact_list, ADD), "Error Constructing TR Edges. Fix This!!!"
            no_human_move = valid_hact_list
        
        if human_action_name != '':
            assert robot_action_name != '', "Error While constructing Human Edge, FIX THIS!!!"
        

        # get the modified robot action name
        mod_raction_name: str = mod_act_dict[robot_action_name]
        robot_move: ADD = self.predicate_sym_map_robot[mod_raction_name]
        _tr_idx: int = self.tr_action_idx_map.get(robot_action_name)

        if human_action_name != '':
            # get the modified human action name
            mod_haction_name: str = mod_act_dict[human_action_name]
            
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name]).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_str_state} --- {robot_action_name} & {human_action_name}---> {next_str_state}")
                
                self.mono_tr_bdd |= curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name]
        else:
            if 'debug' in kwargs:
                edge_exist: bool = (self.mono_tr_bdd & curr_state_sym & robot_move & no_human_move).isZero()
                
                if not edge_exist:
                    print(f"Nondeterminism due to Human Action: {curr_str_state} ---{robot_action_name}---> {next_str_state}")

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
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & self.predicate_sym_map_human[mod_haction_name]
                    
                # if human does not intervene then the edge looks like (robot-action) & not(valid human moves)
                else:
                    self.sym_tr_actions[_tr_idx][_state_idx] |= curr_state_sym & robot_move & no_human_move
                    
            
            elif var == 2 and self.manager.addVar(_idx) in kwargs['prod_curr_list']:
                warnings.warn("Encountered an ambiguous variable during TR construction. FIX THIS!!!")
                sys.exit(-1)
        
        if human_action_name != '':
            assert self.adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get(mod_haction_name) is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.adj_map[curr_state_tuple][mod_raction_name][mod_haction_name] = next_state_tuple

            assert self.org_adj_map.get(curr_state_tuple, {}).get(robot_action_name, {}).get(human_action_name) is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.org_adj_map[curr_state_tuple][robot_action_name][human_action_name] = next_state_tuple
        else:
            assert self.adj_map.get(curr_state_tuple, {}).get(mod_raction_name, {}).get('r') is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.adj_map[curr_state_tuple][mod_raction_name]['r'] = next_state_tuple

            # create another adj map with the edges are stored accoridng to the org name of the action - used during Graph of Utility construction
            assert self.org_adj_map.get(curr_state_tuple, {}).get(robot_action_name, {}).get('r') is None, "Error Computing Adj Dictionary, Fix this!!!"
            self.org_adj_map[curr_state_tuple][robot_action_name]['r'] = next_state_tuple
        
        # update edge count 
        self.ecount += 1


    def _convert_state_lbl_cube_to_func(self, dd_func: ADD, prod_curr_list = None) ->  List[ADD]:
        """
         A helper function to extract a cubes from the DD and print them in human-readable form. 
        """
        tmp_dd_func: BDD = dd_func.bddPattern()
        tmp_ts_x_list: List[BDD] = [_avar.bddPattern() for _avar in prod_curr_list]
        ddVars = []
        for cube in tmp_dd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if var == 2 and self.manager.bddVar(_idx) not in tmp_ts_x_list:
                    continue   # skipping over prime states 
                else:
                    if var == 2:
                        _amb_var.append([self.manager.bddVar(_idx), ~self.manager.bddVar(_idx)])
                    elif var == 0:
                        var_list.append(~self.manager.bddVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.bddVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integer assignment. FIX THIS!!")
                        sys.exit(-1)
                
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    ddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                ddVars.append(reduce(lambda a, b: a & b, var_list))
        
        # convert them back ADDs and return them
        cubes: List[ADD] = [cube.toADD() for cube in ddVars] 
        return cubes                



    def create_transition_system_franka(self,
                                        boxes: List[str],
                                        mod_act_dict,
                                        add_exist_constr: bool = True,
                                        verbose: bool = False,
                                        plot: bool = False,
                                        **kwargs):
        """
         This method overrides the parent method. For every robot action, we loop over all the human actions.
        """
        open_list = defaultdict(lambda: self.manager.addZero())

        closed = self.manager.addZero()

        init_state_sym = self.sym_init_states
        self.sym_state_labels |= init_state_sym
        
        layer = 0

        # creating monolithinc tr bdd to keep track all the transition we are creating to detect nondeterminism
        if 'debug' in kwargs:
            self.mono_tr_bdd = self.manager.addZero() 

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
                sym_state = self._convert_state_lbl_cube_to_func(dd_func=open_list[layer], prod_curr_list=prod_curr_list)
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
                            next_sym_state: ADD = self.get_sym_state_from_exp_states(exp_states=next_exp_state)

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
                                                                        mod_act_dict=mod_act_dict,
                                                                        robot_action_name=raction.name,
                                                                        next_exp_state=next_exp_state,
                                                                        **kwargs)

                            no_human_move_edge: ADD = self.manager.addOne()

                            # if there are any valid human edges from curr state
                            if len(valid_human_edges) > 0:
                                valid_hact: List[ADD] = [self.predicate_sym_map_human[mod_act_dict[ha]] for ha in valid_human_edges]
                                no_human_move_edge: ADD = ~(reduce(lambda x, y: x | y, valid_hact))    

                                assert not no_human_move_edge.isZero(), "Error computing a human no-intervene edge. FIX THIS!!!"        
                    
                            # create robot edges
                            self.add_edge_to_action_tr(robot_action_name=raction.name,
                                                       curr_state_tuple=curr_state_tuple,
                                                       mod_act_dict=mod_act_dict,
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

                            # store the image in the next bucket
                            open_list[layer + 1] |= next_sym_state
                
                layer += 1
        
        if kwargs['print_tr'] is True:
            self._print_plot_tr(plot=plot)
