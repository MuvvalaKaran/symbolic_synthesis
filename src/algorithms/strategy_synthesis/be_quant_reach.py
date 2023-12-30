import time

from collections import defaultdict
from typing import List

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

        # ADD that keeps track of the Adv and Coop optimal values of state at each iteration
        self.adv_winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())
        self.coop_winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())

        # strategy - optimal (state & robot-action) pair stored in the ADD for Adv, Coop games, and BEst-Effort Quant Reachbility games
        self.adv_winning_str: ADD  = self.manager.plusInfinity()
        self.coop_winning_str: ADD = self.manager.plusInfinity()
        self.be_reach_str: ADD = self.manager.plusInfinity()

    def construct_best_effort_reach_strategies(self):
        """
         A function that constructs the best effort reachability strategies for the robot.

         For states (Winning Region) from where these exists a winning strategy, we retain them. For states (Pending Region), from where there is no winning strategy,
          we choose cooperative winning strategy. For the rest of the states (losing region), we choose all valid strategies.  
        """
        c_max: int = self._get_max_tr_action_cost()
        adv_layers_to_conv: int = max(self.adv_game_handle.winning_states.keys())
        coop_layers_to_conv: int = max(self.coop_game_handle.winning_states.keys())

        pending_states: BDD = ~self.adv_winning_states.bddInterval(0, c_max * adv_layers_to_conv) & self.coop_winning_states.bddInterval(0, c_max * coop_layers_to_conv)

        # retain the coop winning strategies for these states
        pending_states_str: ADD = pending_states.toADD() & self.coop_winning_str

        # merge the straegies to construct BE strategies
        self.be_reach_str = self.adv_winning_str | pending_states_str


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
    

    def roll_out_strategy(self, verbose: bool = False, no_intervention: bool = False):
        """
         TODO: This rollout stratgey method is not correct. Will fix it later.
        """
        # BE always exists so we can always roll them out
        self.coop_game_handle.roll_out_strategy(strategy=self.be_reach_str, verbose=verbose, no_intervention=no_intervention)