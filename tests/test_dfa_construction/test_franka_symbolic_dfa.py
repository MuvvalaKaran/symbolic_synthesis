'''
This file tests the LTL/ LTLf to DFA construction. We will test two abstraction construction for 

    1. SymbolicDFAFranka() - A class used to create DFA for the Manipulation examples in MONOLITHIC Fashion. The formulas can only be LTLf!
    2. PartitionedDFA() - A class used to create DFA for the Manipulation example in COMPOSITIONAL Fashion. The formulas can only be LTLf!
'''

import os
import unittest

from cudd import Cudd

from src.symbolic_graphs.graph_search_scripts import FrankaWorld
from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TWO_PLAYER_GAME: bool = True # Set this flag to true when you want to contruct a two-player game env.
TWO_PLAYER_GAME_BND: bool = False  # Set this flag to true when you want to construct som bounded no. off human interventions.


HUMAN_INT_BND: int = 0  # DOES not matter

USE_LTLF: bool = True # Construct DFA from LTLf

DYNAMIC_VAR_ORDERING: bool = False

SUP_LOC = []
TOP_LOC = []

class TestSymbolicFrankaDFAs(unittest.TestCase):
    def test_monolithic_bdd_symbolic_dfa(self):
        """
         Check DFA construction for a single LTLf formula constructed in a Monolithic Fashion using BDD Variables.
        """
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/only_franka_world/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/only_franka_world/problem.pddl"

        formulas = ['F(p00 & p11 & free)']
        
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

        self.assertEqual(len(frankaworld_handle.dfa_x_list), 1,
                             msg=f"Mismatch in the Total # of BDD vars required to construct the Symbolic DFA for formula {formulas[0]}")

    
    def test_composition_bdd_symbolic_dfa(self):
        """
         Check DFA construction for a single LTLf formula constructed in a Compositional Fashion using BDD Variables.
        """
        GAME_ALGORITHM = 'qual'

        formulas = ['F(p00 & p11)']

        domain_file_path = PROJECT_ROOT + "/pddl_files/quantitative_game/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/quantitative_game/problem.pddl"

        cudd_manager = Cudd()

        frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                        problem_file=problem_file_path,
                                                        formulas=formulas,
                                                        manager=cudd_manager,
                                                        sup_locs=SUP_LOC,
                                                        top_locs=TOP_LOC,
                                                        weight_dict=None,
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
        
        self.assertEqual(len(frankapartition_handle.dfa_x_list), 1,
                             msg=f"Mismatch in the Total # of BDD vars required to construct the Symbolic DFA for formula {formulas[0]}")

    def test_monolithic_add_symbolic_dfa(self):
        """
         Check DFA construction for a single LTLf formula constructed in a Monolithic Fashion using ADD Variables.
        """
    
        # Franka World files - No human exists in this only robot world
        domain_file_path = PROJECT_ROOT + "/pddl_files/only_franka_world/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/only_franka_world/problem.pddl"

        formulas = ['F(p00 & p11 & free)']
        
        algo = 'astar'  # 'dijkstras' will work too. 
        
        # grid world dictionary
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

        self.assertEqual(len(frankaworld_handle.dfa_x_list), 1,
                             msg=f"Mismatch in the Total # of BDD vars required to construct the Symbolic DFA for formula {formulas[0]}")


    
    def test_composition_add_symbolic_dfa(self):
        """
         Check DFA construction for a single LTLf formula constructed in a Compositional Fashion using ADD Variables.
        """
        GAME_ALGORITHM = 'quant'
    
        formulas = ['F(p00 & p11)']

        domain_file_path = PROJECT_ROOT + "/pddl_files/quantitative_game/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/quantitative_game/problem.pddl"

        wgt_dict = {
            "transit" : 1,
            "grasp"   : 1,
            "transfer": 1,
            "release" : 1,
            "human": 0
            }

        cudd_manager = Cudd()


        frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                        problem_file=problem_file_path,
                                                        formulas=formulas,
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
        
        self.assertEqual(len(frankapartition_handle.dfa_x_list), 1,
                             msg=f"Mismatch in the Total # of ADD vars required to construct the Symbolic DFA for formula {formulas[0]}")


if __name__ == "__main__":
    unittest.main()