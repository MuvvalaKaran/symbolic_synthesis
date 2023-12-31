import time
import random

from collections import defaultdict
from typing import List, Tuple

from cudd import Cudd, BDD, ADD

from src.algorithms.strategy_synthesis import AdversarialGame, CooperativeGame
from src.symbolic_graphs import ADDPartitionedDFA, DynWeightedPartitionedFrankaAbs


class QuantitativeBestEffortReachSyn(AdversarialGame):
    """
    Class that computes Quantitative Best Effort strategies with reachability objectives. 
    
    The algorithm is as follows:

    1. Given a target set, identify Winning region and synthesize winning strategies.
    2. Given a target set, identify Cooperatively Winning (Pending) region and synthesize cooperative winning strategies
    3. Merge the strategies. 
        3.1 States that belong to winning region play the winning strategy
        3.2 States that belong to pending region play cooperative winning strategy
        3.3 States that belong to losing region play any strategy
    """
    def __init__(self,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 ts_curr_vars: List,
                 dfa_curr_vars: List,
                 ts_obs_vars: List,
                 sys_act_vars: List,
                 env_act_vars: List,
                 cudd_manager: Cudd,
                 monolithic_tr: bool = False):
        super().__init__(ts_handle, dfa_handle, ts_curr_vars, dfa_curr_vars, ts_obs_vars, sys_act_vars, env_act_vars, cudd_manager, monolithic_tr)
        # Adv abd Coop game handles
        self.adv_game_handle: AdversarialGame = None
        self.coop_game_handle: CooperativeGame = None

        # VI dict: ADD that keeps track of the Adv and Coop optimal values of state at each iteration
        self.adv_winning_states: ADD = self.manager.plusInfinity()
        self.coop_winning_states: ADD = self.manager.plusInfinity()
        
        # winning and pending region BDDs
        self.winning_states: BDD = self.manager.bddZero()
        self.pending_states: BDD = self.manager.bddZero()

        # strategy - optimal (state & robot-action) pair stored in the ADD for Adv, Coop games, and BEst-Effort Quant Reachbility games
        self.adv_winning_str: ADD  = self.manager.plusInfinity()
        self.coop_winning_str: ADD = self.manager.plusInfinity()
        self.pending_states_str: ADD = self.manager.plusInfinity()
        self.be_reach_str: ADD = self.manager.plusInfinity()

        # max iteration it took for Adv and Coop games to converge
        self.adv_layers_to_conv: int = 0
        self.coop_layers_to_conv: int = 0
    
    # override parent method
    def is_winning(self) -> bool:
        """
         BE always exists and hence we always return True. We print if an winning (enforcing) or cooperative winning (pending) strategy exists.
        """
        if self.adv_game_handle.is_winning():
            print("Adv Strategy exists!!!!")
        elif self.coop_game_handle.is_winning():
            print("Coop Strategy exists!!!!")
        return True

    def construct_best_effort_reach_strategies(self):
        """
         A function that constructs the best effort reachability strategies for the robot.

         For states (Winning Region) from where these exists a winning strategy, we retain them. For states (Pending Region), from where there is no winning strategy,
          we choose cooperative winning strategy. For the rest of the states (losing region), we choose all valid strategies.  
        """
        c_max: int = self._get_max_tr_action_cost()
        self.adv_layers_to_conv = max(self.adv_game_handle.winning_states.keys())
        self.coop_layers_to_conv = max(self.coop_game_handle.winning_states.keys())

        # sanity checking
        assert self.adv_layers_to_conv > 0 and self.coop_layers_to_conv > 0, "Adv and Coop games VI did not converge correctly. \
              VI Should take atleast 1 iteration to converge. FIX THIS!!!"

        self.pending_states = ~self.adv_winning_states.bddInterval(0, c_max * self.adv_layers_to_conv) & self.coop_winning_states.bddInterval(0, c_max * self.coop_layers_to_conv)

        # retain the coop winning strategies for these states
        self.pending_states_str: ADD = self.coop_winning_str.restrict(self.pending_states.toADD())

        # merge the straegies to construct BE strategies
        self.be_reach_str = self.adv_winning_str | self.pending_states_str


    def solve(self, verbose: bool = False, print_layers: bool = False) -> ADD:
        """
        Overrides the parent class solve method to compute the best effort reachability strategies. 
         First we play adversarial game and then we play cooperative game
        """
        start = time.time()
        self.adv_game_handle = AdversarialGame(ts_handle=self.ts_handle,
                                               dfa_handle=self.dfa_handle,
                                               ts_curr_vars=self.ts_x_list,
                                               dfa_curr_vars=self.dfa_x_list,
                                               ts_obs_vars=self.ts_obs_list,
                                               sys_act_vars=self.sys_act_vars,
                                               env_act_vars=self.env_act_vars,
                                               cudd_manager=self.manager, 
                                               monolithic_tr=self.monolithic_tr)

        self.adv_winning_str = self.adv_game_handle.solve(verbose=verbose, print_layers=print_layers)
        self.adv_winning_states = self.adv_game_handle.winning_states[max(self.adv_game_handle.winning_states.keys())]
        stop = time.time()
        print("Time for solving the Adversarial game: ", stop - start)

        start = time.time()
        self.coop_game_handle = CooperativeGame(ts_handle=self.ts_handle,
                                                dfa_handle=self.dfa_handle,
                                                ts_curr_vars=self.ts_x_list,
                                                dfa_curr_vars=self.dfa_x_list,
                                                ts_obs_vars=self.ts_obs_list,
                                                sys_act_vars=self.sys_act_vars,
                                                env_act_vars=self.env_act_vars,
                                                cudd_manager=self.manager,
                                                monolithic_tr=self.monolithic_tr)
        
        self.coop_winning_str = self.coop_game_handle.solve(verbose=verbose, print_layers=print_layers)
        self.coop_winning_states = self.coop_game_handle.winning_states[max(self.coop_game_handle.winning_states.keys())]

        stop = time.time()
        print("Time for solving the Cooperative game: ", stop - start)

        # finally merge the strategies
        start = time.time()
        self.construct_best_effort_reach_strategies()
        stop = time.time()
        print("Time for constructing the BE strategies: ", stop - start)
    

    def get_strategy_and_val(self, curr_prod_state: ADD, curr_prod_tuple: tuple) -> Tuple[ADD, int]:
        """
         A helper function, that given the current state as 0-1 ADD computes the action to take.
        """
        # first we need to check if the state belong to winning region or pending region
        if curr_prod_state.bddPattern() & self.pending_states:
            opt_sval_cube =  list((curr_prod_state & self.coop_winning_states).generate_cubes())[0]
            curr_prod_state_sval: int = opt_sval_cube[1]
            # this gives us a sval-infinity ADD
            curr_state_act_cubes: ADD =  self.pending_states_str.restrict(curr_prod_state)
        else:
            opt_sval_cube =  list((curr_prod_state & self.adv_winning_states).generate_cubes())[0]
            curr_prod_state_sval: int = opt_sval_cube[1]
            # this gives us a sval-infinity ADD
            curr_state_act_cubes: ADD =  self.adv_winning_str.restrict(curr_prod_state)

        # get the 0-1 version
        act_cube: ADD = curr_state_act_cubes.bddInterval(curr_prod_state_sval, curr_prod_state_sval).toADD()

        if act_cube.isZero():
            print("[Error] Resolve this!")

        if curr_prod_state.bddPattern() & self.pending_states:
            # remove the env vars if state belongs to pending region
            sys_act_cube = act_cube.bddPattern().existAbstract(self.env_cube.bddPattern()).toADD()
            return sys_act_cube, curr_prod_state_sval

        return act_cube, curr_prod_state_sval
    

    def roll_out_strategy(self, verbose: bool = False, ask_usr_input: bool = False, intervene: bool = True):
        """
         A function that rolls out the strategy synthesized by the algorithm.

         NOTE: There are edge cases where BE Strategy get_strategy_and_val() fails to return a valid action. 
         This is because when taking the disjunction, i.e., self.adv_winning_str | self.pending_states_str, sometime the state values in pending region chnages. 
         Will fix this in future commits. For now, a work aorund is extract states values from self.adv_winning_str or self.pending_states_str separately. 
         See get_strategy_and_val() if else condition block for more details. 
        """
        counter = 0
        ract_name: str = ''

        sym_lbl_cubes = self._create_lbl_cubes()

        curr_prod_state = self.init_prod

        if verbose:
            init_ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=self.init_TS, sym_lbl_xcube_list=sym_lbl_cubes)
            init_ts = self.ts_handle.get_state_from_tuple(state_tuple=init_ts_tuple)
            print(f"Init State: {init_ts}")
        
        # until you reach a goal state. . .
        while (self.target_DFA & curr_prod_state).isZero():
            # current state tuple 
            curr_ts_state: ADD = curr_prod_state.existAbstract(self.dfa_xcube & self.sys_env_cube).bddPattern().toADD()   # to get 0-1 ADD
            curr_dfa_state: ADD = curr_prod_state.existAbstract(self.ts_xcube & self.ts_obs_cube & self.sys_env_cube).bddPattern().toADD()
            curr_dfa_tuple: int = self.dfa_sym_to_curr_state_map[curr_dfa_state]
            curr_ts_tuple: tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=curr_ts_state, sym_lbl_xcube_list=sym_lbl_cubes)

            # get the action. . .
            act_cube, curr_prod_state_sval = self.get_strategy_and_val(curr_prod_state=curr_prod_state, curr_prod_tuple=(curr_ts_tuple, curr_dfa_tuple))

            list_act_cube = self.convert_add_cube_to_func(act_cube, curr_state_list=self.sys_act_vars)

            # if multiple winning actions exists from same state
            if len(list_act_cube) > 1:
                ract_name = None
                while ract_name is None:
                    sys_act_cube: List[int] = random.choice(list_act_cube)
                    ract_name = self.ts_sym_to_robot_act_map.get(sys_act_cube, None)
                
            else:
                ract_name: str = self.ts_sym_to_robot_act_map[act_cube]
            
            if verbose:
                print(f"Step {counter}: Conf: {self.ts_handle.get_state_from_tuple(curr_ts_tuple)} Act: {ract_name}")
            
            # look up the next tuple 
            next_exp_states = self.ts_handle.adj_map[curr_ts_tuple][ract_name]['r']
            next_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)

            # Human Intervention
            # flip a coin and choose to intervene or not intervene
            coin = random.randint(0, 1)
            # coin = 1
            if intervene and coin:
                # check if there any human action from the current state
                valid_acts = set(self.ts_handle.adj_map.get(curr_ts_tuple, {}).get(ract_name, {}).keys())
                human_acts = valid_acts.difference('r')
                
                if len(human_acts) > 0:
                    next_tuple = self.human_intervention(ract_name=ract_name,
                                                         curr_state_tuple=curr_ts_tuple,
                                                         rnext_tuple=next_tuple,
                                                         curr_dfa_state=curr_dfa_state,
                                                         valid_human_acts=human_acts,
                                                         verbose=verbose)
            
            # look up its corresponding formula
            curr_ts_state: ADD = self.ts_handle.get_sym_state_from_tuple(next_tuple)

            # convert the 0-1 ADD to BDD for DFA edge checking
            curr_ts_lbl: BDD = curr_ts_state.existAbstract(self.ts_xcube).bddPattern()

            # create DFA edge and check if it satisfies any of the edges or not
            for dfa_state in self.dfa_sym_to_curr_state_map.keys():
                bdd_dfa_state: BDD = dfa_state.bddPattern()
                dfa_pre: BDD = bdd_dfa_state.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
                edge_exists: bool = not (dfa_pre & (curr_dfa_state.bddPattern() & curr_ts_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state: BDD = dfa_state
                    break
            
            curr_prod_state: ADD = curr_ts_state & curr_dfa_state

            counter += 1
