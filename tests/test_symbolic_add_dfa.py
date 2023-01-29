'''
 This file tests the LTL/ LTLf to DFA construction. We will test three abstarction construction for 

    1. SymboliDFA() - A class used to create the DFA for the 2d gridworld exmaples. The formulas can be LTL or LTLf!
    2. SymbolicDFAFranka() - A class used to create DFA for the Manipulation examples in MONOLITHIC Fashion. The formulas can be LTL or LTLf!
    3. PartitionedDFA() - A class used to create DFA for the Manipulation example in COMPOSITIONAL Fashion. The formula can be only be LTLf!
'''
import os
import unittest

from cudd import Cudd

from src.symbolic_graphs.graph_search_scripts import SimpleGridWorld

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DYNAMIC_VAR_ORDERING: bool = False
GRID_WORLD_SIZE: int = 5
DIJKSTRAS: bool = True  # set this flag to true when you want to use Dijkstras
ASTAR: bool = False # set this flag to true when you want to use A* algorithm

# Note on the griworld: The location start from l1 and go up till l25 (n x n - square gridworld).
# We start at the bottom left at l1 and fill the first till l5 and continue on to the next row starting from l6 from the left so and on.

SINGLE_FORMULA = ['F(l21 & F(l5) & F(l25))']  # optimal strategy is to traverse the grid by first visiting l21, then l25 and finally l5.

# 5 state formula for 5x5 GW
# Optimal strategy - due to the inclusion of the last formula.
# The robot first visits l2 and then traverses the gridworld along the boundary in clockwise fashion.
MULTIPLE_FORMULAS = ["F(l21 & F(l5) & F(l25))",
                     "F(l22 & F(l4) & F(l20))",
                     "F(l23 & F(l3) & F(l15))",
                     "F(l24 & F(l2) & F(l10))",
                     "F(l16 & F(l21) & F(l2))",
                     "F(l11 & F(l22) & F(l3))",
                     "F(l6 & F(l23) & F(l4))",
                     "F(l2 & F(l20) & F(l16))"]


class TestLTLSymbolicDFA(unittest.TestCase):
    def test_single_LTL_Symbolic_DFA(self):
        """
         Check DFA construction for a single LTL formula for 2d gridworld robot
        """
        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'

        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/grid_world/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=False,
                                           dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                           verbose=False,
                                           plot_ts=False,
                                           plot_obs=False,
                                           plot=False)
        
        # build the TS and DFA(s)
        gridworld_handle.build_abstraction()

        self.assertEqual(cudd_manager.size(), 21,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic GridWorld Abstraction")

        policy: dict = gridworld_handle.solve(verbose=False)

        # empty dictionary evaluate to False
        self.assertEqual(bool(policy), True,
                         msg="Could not synthesize a stratgy.")

        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTL_str.svg')


    def test_multiple_single_LTL_Symbolic_DFA(self):
        """
         Check DFA construction for multiple LTL formulas for 2d gridworld robot
        """
        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'
        
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/grid_world/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=False,
                                           dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                           verbose=False,
                                           plot_ts=False,
                                           plot_obs=False,
                                           plot=False)
        
        # build the TS and DFA(s)
        gridworld_handle.build_abstraction()

        self.assertEqual(cudd_manager.size(), 63,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic GridWorld Abstraction")

        policy: dict = gridworld_handle.solve(verbose=False)
        
        # empty dictionary evaluate to False
        self.assertEqual(bool(policy), True,
                         msg="Could not synthesize a stratgy.")
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTL_str.svg')

    def test_single_LTLf_Symbolic_DFA(self):
        """
         Check DFA construction for a single LTLf formulas for 2d gridworld robot
        """
        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'
        
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/grid_world/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=True,
                                           dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                           verbose=False,
                                           plot_ts=False,
                                           plot_obs=False,
                                           plot=False)
        
        # build the TS and DFA(s)
        gridworld_handle.build_abstraction()

        self.assertEqual(cudd_manager.size(), 21,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic GridWorld Abstraction")

        policy: dict = gridworld_handle.solve(verbose=False)

        # empty dictionary evaluate to False
        self.assertEqual(bool(policy), True,
                         msg="Could not synthesize a stratgy.")
                         
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTLf_str.svg')

    def test_multiple_single_LTLf_Symbolic_DFA(self):
        """
         Check DFA construction for multiple LTLf formulas for 2d gridworld robot
        """
        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'
        
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/grid_world/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=True,
                                           dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                           verbose=False,
                                           plot_ts=False,
                                           plot_obs=False,
                                           plot=False)
        
        # build the TS and DFA(s)
        gridworld_handle.build_abstraction()

        self.assertEqual(cudd_manager.size(), 63,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic GridWorld Abstraction")

        policy: dict = gridworld_handle.solve(verbose=False)

        # empty dictionary evaluate to False
        self.assertEqual(bool(policy), True,
                         msg="Could not synthesize a stratgy.")

        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTLf_str.svg')


if __name__ == "__main__":
    unittest.main()