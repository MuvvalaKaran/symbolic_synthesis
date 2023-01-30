'''
 This file runs all Manipulation examples with LTLf formulas. The pddl file only describes the robot operating in the workspace. 
'''
import os
import unittest

from typing import List
from cudd import Cudd

from src.symbolic_graphs.graph_search_scripts import FrankaWorld

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

USE_LTLF: bool = True # Construct DFA from LTLf

DYNAMIC_VAR_ORDERING: bool = False

SUP_LOC = []
TOP_LOC = []

# NOTE: When writing formulas for FrankaWorld we need to have gripper status in the formulas. 

class TestOnlyFranka(unittest.TestCase):
    def test_only_franka_single_bfs(self):
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"

        formulas = ['F(p01 & free  & XF(p17 & free))',    # tricky complicated formula
                    'F(p01 & p16 & free & F(p17 & free & F(p16 & free)))',  # neat complicated formula
                    'F((p01 | p11) & free)',
                    'F(p00 & p12 & p21 & free) & G(~(p00 & p21 & free) -> ~(p12 & free))',  # arch construction - fixed supports
                    "(F((((p00 & p21) | (p01 & p20)) & p12 & free)) & G((!(((p00 & p21 & free) | (p01 & p20 & free))) -> !(p12 & free))))"]  # flexible support boxes


        algo = 'bfs'

        # correct values
        cor_total_vars: List[int] = [38, 38, 36, 38, 38]
        
        for task_id, task in enumerate(formulas):

            cudd_manager = Cudd()

            # frankaworld stuff
            frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                            problem_file=problem_file_path,
                                            formulas=[task],
                                            manager=cudd_manager,
                                            sup_locs=SUP_LOC,
                                            top_locs=TOP_LOC,
                                            weight_dict=None,
                                            ltlf_flag=USE_LTLF,
                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                            algorithm=algo,
                                            verbose=False,
                                            plot_ts=False,
                                            plot_obs=False,
                                            plot=False)

            # build the abstraction
            frankaworld_handle.build_abstraction()

            total_vars: int = cudd_manager.size()
            
            self.assertEqual(total_vars,
                             cor_total_vars[task_id],
                             msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Abstraction for formula {task}")

            # compute the policy
            policy: dict = frankaworld_handle.solve(verbose=False)

            # empty dictionary evaluates to False
            self.assertEqual(bool(policy), True,
                            msg="Could not synthesize a stratgy.")

            frankaworld_handle.simulate(action_dict=policy, print_strategy=False)


    def test_only_franka_single_astar(self):
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"

        formulas = ['F(p01 & free  & XF(p17 & free))',    # tricky complicated formula
                    'F(p01 & p16 & free & F(p17 & free & F(p16 & free)))',  # neat complicated formula
                    'F((p01 | p11) & free)',
                    'F(p00 & p12 & p21 & free) & G(~(p00 & p21 & free) -> ~(p12 & free))',  # arch construction - fixed supports
                    "(F((((p00 & p21) | (p01 & p20)) & p12 & free)) & G((!(((p00 & p21 & free) | (p01 & p20 & free))) -> !(p12 & free))))"]  # flexible support boxes

        wgt_dict = {
            "transit" : 1,
            "grasp"   : 2,
            "transfer": 3,
            "release" : 4,
            }


        algo = 'astar'

        # correct values
        cor_total_vars: List[int] = [38, 38, 36, 38, 38]
        
        for task_id, task in enumerate(formulas):

            cudd_manager = Cudd()

            # frankaworld stuff
            frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                            problem_file=problem_file_path,
                                            formulas=[task],
                                            manager=cudd_manager,
                                            sup_locs=SUP_LOC,
                                            top_locs=TOP_LOC,
                                            weight_dict=wgt_dict,
                                            ltlf_flag=USE_LTLF,
                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                            algorithm=algo,
                                            verbose=False,
                                            plot_ts=False,
                                            plot_obs=False,
                                            plot=False)

            # build the abstraction
            frankaworld_handle.build_abstraction()

            total_vars: int = cudd_manager.size()
            
            self.assertEqual(total_vars,
                             cor_total_vars[task_id],
                             msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Abstraction for formula {task}")

            # compute the policy
            policy: dict = frankaworld_handle.solve(verbose=False)

            # empty dictionary evaluates to False
            self.assertEqual(bool(policy), True,
                            msg="Could not synthesize a stratgy.")

            frankaworld_handle.simulate(action_dict=policy, print_strategy=False)

    def test_only_franka_single_dijkstras(self):
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"

        formulas = ['F(p01 & free  & XF(p17 & free))',    # tricky complicated formula
                    'F(p01 & p16 & free & F(p17 & free & F(p16 & free)))',  # neat complicated formula
                    'F((p01 | p11) & free)',
                    'F(p00 & p12 & p21 & free) & G(~(p00 & p21 & free) -> ~(p12 & free))',  # arch construction - fixed supports
                    "(F((((p00 & p21) | (p01 & p20)) & p12 & free)) & G((!(((p00 & p21 & free) | (p01 & p20 & free))) -> !(p12 & free))))"]  # flexible support boxes

        wgt_dict = {
            "transit" : 1,
            "grasp"   : 2,
            "transfer": 3,
            "release" : 4,
            }


        algo = 'dijkstras'

        # correct values
        cor_total_vars: List[int] = [38, 38, 36, 38, 38]
        
        for task_id, task in enumerate(formulas):

            cudd_manager = Cudd()

            # frankaworld stuff
            frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                            problem_file=problem_file_path,
                                            formulas=[task],
                                            manager=cudd_manager,
                                            sup_locs=SUP_LOC,
                                            top_locs=TOP_LOC,
                                            weight_dict=wgt_dict,
                                            ltlf_flag=USE_LTLF,
                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                            algorithm=algo,
                                            verbose=False,
                                            plot_ts=False,
                                            plot_obs=False,
                                            plot=False)

            # build the abstraction
            frankaworld_handle.build_abstraction()

            total_vars: int = cudd_manager.size()
            
            self.assertEqual(total_vars,
                             cor_total_vars[task_id],
                             msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Abstraction for formula {task}")

            # compute the policy
            policy: dict = frankaworld_handle.solve(verbose=False)

            # empty dictionary evaluates to False
            self.assertEqual(bool(policy), True,
                            msg="Could not synthesize a stratgy.")

            frankaworld_handle.simulate(action_dict=policy, print_strategy=False)

    def test_only_franka_multiple_bfs(self):
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"

        # the synthesize strategy partially completes task1 then, proceeds to finish task2 and then finishes task1.
         
        # The policy is searched over the product space of all the tasks
        formulas = ['F(p01 & free  & XF(p17 & free))',
                    "(F((((p00 & p21) | (p01 & p20)) & p12 & free)) & G((!(((p00 & p21 & free) | (p01 & p20 & free))) -> !(p12 & free))))"]


        algo = 'bfs'

        cudd_manager = Cudd()

        # frankaworld stuff
        frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                        problem_file=problem_file_path,
                                        formulas=formulas,
                                        manager=cudd_manager,
                                        sup_locs=SUP_LOC,
                                        top_locs=TOP_LOC,
                                        weight_dict=None,
                                        ltlf_flag=USE_LTLF,
                                        dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                        algorithm=algo,
                                        verbose=False,
                                        plot_ts=False,
                                        plot_obs=False,
                                        plot=False)

        # build the abstraction
        frankaworld_handle.build_abstraction()

        total_vars: int = cudd_manager.size()
        
        self.assertEqual(total_vars, 42,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Abstraction for formulad {formulas}")

        # compute the policy
        policy: dict = frankaworld_handle.solve(verbose=False)

        # empty dictionary evaluates to False
        self.assertEqual(bool(policy), True,
                        msg="Could not synthesize a stratgy.")

        frankaworld_handle.simulate(action_dict=policy, print_strategy=False)

    def test_only_franka_multiple_astar(self):
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"

        # the synthesize strategy partially completes task1 then, proceeds to finish task2 and then finishes task1.
         
        # The policy is searched over the product space of all the tasks
        formulas = ['F(p01 & free  & XF(p17 & free))',
                    "(F((((p00 & p21) | (p01 & p20)) & p12 & free)) & G((!(((p00 & p21 & free) | (p01 & p20 & free))) -> !(p12 & free))))"]

        algo = 'astar'

        wgt_dict = {
            "transit" : 1,
            "grasp"   : 2,
            "transfer": 3,
            "release" : 4,
            }

        cudd_manager = Cudd()

        # frankaworld stuff
        frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                        problem_file=problem_file_path,
                                        formulas=formulas,
                                        manager=cudd_manager,
                                        sup_locs=SUP_LOC,
                                        top_locs=TOP_LOC,
                                        weight_dict=wgt_dict,
                                        ltlf_flag=USE_LTLF,
                                        dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                        algorithm=algo,
                                        verbose=False,
                                        plot_ts=False,
                                        plot_obs=False,
                                        plot=False)

        # build the abstraction
        frankaworld_handle.build_abstraction()

        total_vars: int = cudd_manager.size()
        
        self.assertEqual(total_vars, 42,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Abstraction for formulad {formulas}")

        # compute the policy
        policy: dict = frankaworld_handle.solve(verbose=False)

        # empty dictionary evaluates to False
        self.assertEqual(bool(policy), True,
                        msg="Could not synthesize a stratgy.")

        frankaworld_handle.simulate(action_dict=policy, print_strategy=False)

    def test_only_franka_multiple_dijkstras(self):
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"

        # the synthesize strategy partially completes task1 then, proceeds to finish task2 and then finishes task1.
         
        # The policy is searched over the product space of all the tasks
        formulas = ['F(p01 & free  & XF(p17 & free))',
                    "(F((((p00 & p21) | (p01 & p20)) & p12 & free)) & G((!(((p00 & p21 & free) | (p01 & p20 & free))) -> !(p12 & free))))"]

        algo = 'dijkstras'

        wgt_dict = {
            "transit" : 1,
            "grasp"   : 2,
            "transfer": 3,
            "release" : 4,
            }

        cudd_manager = Cudd()

        # frankaworld stuff
        frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                        problem_file=problem_file_path,
                                        formulas=formulas,
                                        manager=cudd_manager,
                                        sup_locs=SUP_LOC,
                                        top_locs=TOP_LOC,
                                        weight_dict=wgt_dict,
                                        ltlf_flag=USE_LTLF,
                                        dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                        algorithm=algo,
                                        verbose=False,
                                        plot_ts=False,
                                        plot_obs=False,
                                        plot=False)

        # build the abstraction
        frankaworld_handle.build_abstraction()

        total_vars: int = cudd_manager.size()
        
        self.assertEqual(total_vars, 42,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Abstraction for formulad {formulas}")

        # compute the policy
        policy: dict = frankaworld_handle.solve(verbose=False)

        # empty dictionary evaluates to False
        self.assertEqual(bool(policy), True,
                        msg="Could not synthesize a stratgy.")

        frankaworld_handle.simulate(action_dict=policy, print_strategy=False)


if __name__ == "__main__":
    unittest.main()