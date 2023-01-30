'''
 This file test the gridworld agent strategy synthesis for single formula and multiple forumals with (dijkstras, A*) and without edge weights. 
'''

import os
import unittest

from cudd import Cudd

from src.symbolic_graphs.graph_search_scripts import SimpleGridWorld

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

GRID_WORLD_SIZE: int = 5

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


class TestGridWorld(unittest.TestCase):
    def test_dynamic_variable_ordering(self):
        """
         Test Gridworld str synthesis with dynamic variable ordering
        """
        algo = 'bfs'

        # initiate a manager
        cudd_manager = Cudd()

        DYNAMIC_VAR_ORDERING: bool = True

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=None,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTL_bfs_str.svg')

    ######################################################################
    ###################### LTL GRIDWORLD FORMULAS ########################
    ######################################################################
    def test_single_ltl_bfs(self):
        """
         Test gridworld strategy synthesis using BDD variables in BFS fashion for single LTL formula
        """
        algo = 'bfs'

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=None,
                                           ltlf_flag=False,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTL_bfs_str.svg')


    def test_single_ltl_astar(self):
        """
         Test gridworld strategy synthesis using ADD variables using A* for single LTL formula
        """
        algo = 'astar'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=False,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTL_astar_str.svg')


    def test_single_ltl_dijkstras(self):
        """
         Test gridworld strategy synthesis using ADD variables using Dijkstras for single LTL formula
        """
        algo = 'dijkstras'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=False,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTL_dijkstras_str.svg')
    
    
    def test_multiple_ltl_bfs(self):
        """
         Test gridworld strategy synthesis using BDD variables in BFS fashion for multiple LTL formulas
        """
        algo = 'bfs'

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=None,
                                           ltlf_flag=False,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTL_bfs_str.svg')


    def test_multiple_ltl_astar(self):
        """
         Test gridworld strategy synthesis using ADD variables using A* for multiple LTL formulas
        """
        algo = 'astar'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=False,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTL_astar_str.svg')


    def test_multiple_ltl_dijkstras(self):
        """
         Test gridworld strategy synthesis using ADD variables using Dijkstras for multiple LTL formulas
        """
        algo = 'dijkstras'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=False,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTL_dijkstras_str.svg')
    

    ######################################################################
    ##################### LTLf GRIDWORLD FORMULAS ########################
    ######################################################################

    def test_single_ltlf_bfs(self):
        """
         Test gridworld strategy synthesis using BDD variables in BFS fashion for single LTLf formula
        """
        algo = 'bfs'

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=None,
                                           ltlf_flag=True,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTLf_bfs_str.svg')

    def test_single_ltlf_astar(self):
        """
         Test gridworld strategy synthesis using ADD variables using A* for single LTLf formula
        """
        algo = 'astar'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=True,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTLf_astar_str.svg')

    def test_single_ltlf_dijkstras(self):
        """
         Test gridworld strategy synthesis using ADD variables using Dijkstras for single LTLf formula
        """
        algo = 'dijkstras'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=SINGLE_FORMULA,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=False,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_LTLf_dijkstras_str.svg')
    
    def test_multiple_ltlf_bfs(self):
        """
         Test gridworld strategy synthesis using BDD variables in BFS fashion for multiple LTLf formulas
        """
        algo = 'bfs'

        # initiate a manager
        cudd_manager = Cudd()

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=None,
                                           ltlf_flag=True,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTLf_bfs_str.svg')


    def test_multiple_ltlf_astar(self):
        """
         Test gridworld strategy synthesis using ADD variables using A* for multiple LTLf formulas
        """
        algo = 'astar'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=True,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTLf_astar_str.svg')

    def test_multiple_ltlf_dijkstras(self):
        """
         Test gridworld strategy synthesis using ADD variables using Dijkstras for multiple LTLf formulas
        """
        algo = 'dijkstras'

        # initiate a manager
        cudd_manager = Cudd()

        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + f"/pddl_files/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"

        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=MULTIPLE_FORMULAS,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=True,
                                           dyn_var_ord=False,
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
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE, file_name='test_nLTLf_dijkstras_str.svg')


if __name__ == "__main__":
    unittest.main()