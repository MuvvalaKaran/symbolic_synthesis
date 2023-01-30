import os
import unittest

from typing import List
from cudd import Cudd, ADD

from src.algorithms.strategy_synthesis import CooperativeGame
from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld



# config flags 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TWO_PLAYER_GAME: bool = True # Set this flag to true when you want to contruct a two-player game env.
TWO_PLAYER_GAME_BND: bool = False  # Set this flag to true when you want to construct som bounded no. off human interventions.
GAME_ALGORITHM = 'quant-coop' # choose qual for qualitative game, quant-adv for quantitative adversarial game, and quant-coop for cooperative game

HUMAN_INT_BND: int = 0  # DOES not matter

USE_LTLF: bool = True # Construct DFA from LTLf

DYNAMIC_VAR_ORDERING: bool = False

SUP_LOC = []
TOP_LOC = []


class TestCoopGame(unittest.TestCase):
    def test_abstraction(self):
        """
         Check all the tests related abstraction construction
        """
        # TEST for various formulas 
        formulas = ['F(p00 & p11)', 
                    'F(p01 & XF(p17))',
                    'F(p11 & p06 & F(p07 & F(p06)))',
                    'F(p01 | p11)']

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"


        wgt_dict = {
            "transit" : 1,
            "grasp"   : 1,
            "transfer": 1,
            "release" : 1,
            "human": 0
            }
        
        # correct values
        cor_total_vars: List[int] = [16, 17, 17, 16]

        for task_id, task in enumerate(formulas):
            cudd_manager = Cudd()


            frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                            problem_file=problem_file_path,
                                                            formulas=[task],
                                                            manager=cudd_manager,
                                                            sup_locs=SUP_LOC,
                                                            top_locs=TOP_LOC,
                                                            weight_dict=wgt_dict,
                                                            ltlf_flag=USE_LTLF,
                                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                            algorithm=GAME_ALGORITHM,
                                                            verbose=False,
                                                            plot_ts=False,
                                                            plot_obs=False,
                                                            plot=False)
            
            ### ENSURE that the support locations are empty in the actual config file - else this might throw errors!
            # build the abstraction
            frankapartition_handle.build_abstraction(dynamic_env=TWO_PLAYER_GAME,
                                                    bnd_dynamic_env=TWO_PLAYER_GAME_BND,
                                                    max_human_int=HUMAN_INT_BND)

            total_vars: int = cudd_manager.size()
            
            self.assertEqual(total_vars,
                             cor_total_vars[task_id],
                             msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Weighted Abstraction for formula {task}")
            
            self.assertEqual(frankapartition_handle.ts_handle.ecount,
                             394,
                             msg=f"Mismatch in the # of edges in the Symbolic Weighted Abstraction for formula {task}.")
    

    def test_synthesis(self):
        """
         Check all the tests related Quantitative strategy synthesis under quantitative constraints. 
        """

        # TEST for various formulas 
        formulas = ['F(p00 & p11)',    # Adv strategy shoould not exist
                    'F(p01 & XF(p17))',  # Adv exists as the robot can force the human
                    'F(p11 & p06 & F(p07 & F(p06)))',  # Adv. strategy exisits. In coop setting the human will help in satisfying the inner formula
                    'F(p01 | p11)']  # Adv. strategy will prefer p01 and p11 is not possible

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"

        wgt_dict = {
            "transit" : 1,
            "grasp"   : 1,
            "transfer": 1,
            "release" : 1,
            "human": 0
            }

        # No. of iteration req. to reach the fixed point
        cor_fp: List[int] = [13, 13, 13, 5]

        # Min. energy required
        corr_eng: List[int] = [8, 5, 10, 4]

        for task_id, task in enumerate(formulas):
            cudd_manager = Cudd()
            frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                            problem_file=problem_file_path,
                                                            formulas=[task],
                                                            manager=cudd_manager,
                                                            sup_locs=SUP_LOC,
                                                            top_locs=TOP_LOC,
                                                            weight_dict=wgt_dict,
                                                            ltlf_flag=USE_LTLF,
                                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                            algorithm=GAME_ALGORITHM,
                                                            verbose=False,
                                                            plot_ts=False,
                                                            plot_obs=False,
                                                            plot=False)
            # build the abstraction
            frankapartition_handle.build_abstraction(dynamic_env=TWO_PLAYER_GAME,
                                                    bnd_dynamic_env=TWO_PLAYER_GAME_BND,
                                                    max_human_int=HUMAN_INT_BND)
            
            min_min_handle = CooperativeGame(ts_handle=frankapartition_handle.ts_handle,
                                                dfa_handle=frankapartition_handle.dfa_handle,
                                                ts_curr_vars=frankapartition_handle.ts_x_list,
                                                dfa_curr_vars=frankapartition_handle.dfa_x_list,
                                                ts_obs_vars=frankapartition_handle.ts_obs_list,
                                                sys_act_vars=frankapartition_handle.ts_robot_vars,
                                                env_act_vars=frankapartition_handle.ts_human_vars,
                                                cudd_manager=frankapartition_handle.manager)

            win_str: ADD = min_min_handle.solve(verbose=False)

            # ensure winning strategy exisits
            self.assertNotEqual(win_str, cudd_manager.addZero(), "Could not synthesize a winning strategy for adversarial game")

            if win_str:
                # ensure that you reach the fixed point correctly
                self.assertEqual(max(min_min_handle.winning_states.keys()), cor_fp[task_id], "Error computing the fixed point.")

                init_state_cube = list(((min_min_handle.init_TS & min_min_handle.init_DFA) & min_min_handle.winning_states[cor_fp[task_id]]).generate_cubes())[0]
                init_val: int = init_state_cube[1]

                # ensure that the energy required is correct
                self.assertEqual(init_val, corr_eng[task_id], "Error in the minimum energy required the task under adv. env. assumption.")

                # this has to be done to ensure that
                # 1) the strategy synthesized does indeed reach the accepting state, and
                # 2) to ensure that the code does not seg fault.
                min_min_handle.roll_out_strategy(strategy=win_str, verbose=False)
        


if __name__ == "__main__":
    unittest.main()