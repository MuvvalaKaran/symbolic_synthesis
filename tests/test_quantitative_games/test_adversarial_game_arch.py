import os
import unittest

from typing import List
from cudd import Cudd, ADD

from src.algorithms.strategy_synthesis import AdversarialGame
from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld

# config flags 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TWO_PLAYER_GAME: bool = True # Set this flag to true when you want to contruct a two-player game env.
TWO_PLAYER_GAME_BND: bool = False  # Set this flag to true when you want to construct som bounded no. off human interventions.
GAME_ALGORITHM = 'quant-coop' # choose qual for qualitative game, quant-adv for quantitative adversarial game, and quant-coop for cooperative game

HUMAN_INT_BND: int = 0  # DOES not matter

USE_LTLF: bool = True # Construct DFA from LTLf

DYNAMIC_VAR_ORDERING: bool = False

SUP_LOC = ['l0', 'l1']   # support for Arch
TOP_LOC = ['l2']    


class TestAdversarialGameArch(unittest.TestCase):
    def test_abstraction(self):
        """
         Check all the tests related Arch abstraction construction
        """
        
        formulas = ['F(p00 & p12 & p21) & G(~(p00 & p21) -> ~(p12))']

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_paths = [
            PROJECT_ROOT + "/pddl_files/problem_arch1.pddl",   # all boxed are withing robot's reach
            PROJECT_ROOT + "/pddl_files/problem_arch2.pddl"]    # robot cannot force human to build the arch


        wgt_dict = {
            "transit" : 1,
            "grasp"   : 1,
            "transfer": 1,
            "release" : 1,
            "human": 0
            }
        
        # correct values
        cor_total_vars: List[int] = [22, 23]
        cor_ecount: List[int] = [4884, 10188]

        for arch_id, arch_problem in enumerate(problem_file_paths):
            cudd_manager = Cudd()


            frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                            problem_file=arch_problem,
                                                            formulas=formulas,
                                                            sup_locs=SUP_LOC,
                                                            top_locs=TOP_LOC,
                                                            manager=cudd_manager,
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
                             cor_total_vars[arch_id],
                             msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Weighted Abstraction for formula {formulas[0]}")
            
            self.assertEqual(frankapartition_handle.ts_handle.ecount,
                             cor_ecount[arch_id],
                             msg=f"Mismatch in the # of edges in the Symbolic Weighted Abstraction for formula {formulas[0]}.")
    

    def test_synthesis(self):
        """
         Check all the tests related Arch Quantitative strategy synthesis under quantitative constraints. 
        """
        formulas = ['F(p00 & p12 & p21) & G(~(p00 & p21) -> ~(p12))']


        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_paths = [
            PROJECT_ROOT + "/pddl_files/problem_arch1.pddl",   # all boxes are within robot's reach
            PROJECT_ROOT + "/pddl_files/problem_arch2.pddl"]    # robot cannot force human to build the arch

        wgt_dict = {
            "transit" : 1,
            "grasp"   : 1,
            "transfer": 1,
            "release" : 1,
            "human": 0
            }

        # No. of iteration req. to reach the fixed point
        cor_fp: List[int] = [17, 3]

        # Min. energy required
        corr_eng: List[int] = [12, None]

        for arch_id, arch_problem in enumerate(problem_file_paths):
            cudd_manager = Cudd()
            frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                            problem_file=arch_problem,
                                                            formulas=formulas,
                                                            sup_locs=SUP_LOC,
                                                            top_locs=TOP_LOC,
                                                            manager=cudd_manager,
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
            
            min_max_handle = AdversarialGame(ts_handle=frankapartition_handle.ts_handle,
                                                dfa_handle=frankapartition_handle.dfa_handle,
                                                ts_curr_vars=frankapartition_handle.ts_x_list,
                                                dfa_curr_vars=frankapartition_handle.dfa_x_list,
                                                ts_obs_vars=frankapartition_handle.ts_obs_list,
                                                sys_act_vars=frankapartition_handle.ts_robot_vars,
                                                env_act_vars=frankapartition_handle.ts_human_vars,
                                                cudd_manager=frankapartition_handle.manager)

            win_str: ADD = min_max_handle.solve(verbose=False)

            # ensure winning strategy exisits
            self.assertNotEqual(win_str, cudd_manager.addZero(), "Could not synthesize a winning strategy for adversarial game")

            if win_str:
                # ensure that you reach the fixed point correctly
                self.assertEqual(max(min_max_handle.winning_states.keys()), cor_fp[arch_id], "Error computing the fixed point.")

                init_state_cube = list(((min_max_handle.init_TS & min_max_handle.init_DFA) & min_max_handle.winning_states[cor_fp[arch_id]]).generate_cubes())[0]
                init_val: int = init_state_cube[1]

                # ensure that the energy required is correct
                self.assertEqual(init_val, corr_eng[arch_id], "Error in the minimum energy required the task under adv. env. assumption.")

                # this has to be done to ensure that
                # 1) the strategy synthesized does indeed reach the accepting state, and
                # 2) to ensure that the code does not seg fault.
                min_max_handle.roll_out_strategy(strategy=win_str, verbose=False)
        


if __name__ == "__main__":
    unittest.main()
