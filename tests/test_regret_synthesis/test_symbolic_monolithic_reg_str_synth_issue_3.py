'''
 This files tests all steps executed during synthesis of regret-minimizing strategies. 
 We override explicit Graph of Utility construction with symbolic Graph of Utility construction methods.
'''

import os
import unittest

from typing import List
from cudd import Cudd, ADD

from src.symbolic_graphs.strategy_synthesis_scripts import FrankaSymbolicRegretSynthesis

from src.symbolic_graphs.hybrid_regret_graphs import HybridGraphOfBR

from src.algorithms.strategy_synthesis import SymbolicGraphOfUtlCooperativeGame

from .test_hybrid_monolithic_reg_str_synth_issue_3 import TestMonoRegretStrSynth

# config flags 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DYNAMIC_VAR_ORDERING: bool = False

USE_LTLF: bool = True # Construct DFA from LTLf

MONOLITHIC_TR: bool = False

SUP_LOC = []
TOP_LOC = []    


class TestMonoSymbolicRegretStrSynth(TestMonoRegretStrSynth):
    """
     We override the Graph of Utility functionality to construct it purely symbolically.
    """
    @classmethod
    def setUpClass(cls) -> None:
        cls.formulas = ['F(p01) | F(p17)']

        domain_file_path: str = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path: str = PROJECT_ROOT + "/pddl_files/problem_1.pddl"

        wgt_dict = {
                "transit" : 1,
                "grasp"   : 1,
                "transfer": 1,
                "release" : 1,
                "human": 0
                }

        cls.cudd_manager = Cudd()

        cls.regret_synthesis_handle = FrankaSymbolicRegretSynthesis(domain_file=domain_file_path,
                                                                    problem_file=problem_file_path,
                                                                    formulas=cls.formulas,
                                                                    manager=cls.cudd_manager,
                                                                    sup_locs=SUP_LOC,
                                                                    top_locs=TOP_LOC,
                                                                    weight_dict=wgt_dict,
                                                                    ltlf_flag=USE_LTLF,
                                                                    dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                                    weighting_factor=3,
                                                                    reg_factor=1.25,
                                                                    algorithm=None,
                                                                    verbose=False,
                                                                    plot_ts=False,
                                                                    print_layer=False,
                                                                    plot_obs=False,
                                                                    plot=False)
        
        cls.regret_synthesis_handle.build_abstraction()

    def test_3_graph_of_utility_constrcution(self):
        """
         Test Symbolic Graph of Utility (unrolling original graph) Construction
        """
        self.regret_synthesis_handle.build_add_graph_of_utility(verbose=False, just_adv_game=False, monolithic_tr=True)

        # verify # of states; # of leaf nodes required to constuct the graph.
        no_of_states: int = self.regret_synthesis_handle.graph_of_utls_handle.scount
        no_of_leaf_nodes: int = self.regret_synthesis_handle.graph_of_utls_handle.lcount

        leaf_values: int = self.regret_synthesis_handle.graph_of_utls_handle.leaf_vals
        fp_layer: int =  max(self.regret_synthesis_handle.graph_of_utls_handle.open_list.keys())
        
        self.assertEqual(no_of_states, 12, "Error constructing Graph of Utility. Mismatch in # of states")
        self.assertEqual(no_of_leaf_nodes, 4, "Error constructing Graph of Utility. Mismatch in # of Leaf nodes")

        self.assertEqual(leaf_values, set([1, 3, 4]), "Error constructing Graph of Utility. Mismatch in # utility values associated with leaf nodes")
        self.assertEqual(fp_layer, 8, "Error constructing Graph of Utility. Mismatch in # of layers required to construct the graph")
        

    def test_4_cVal_computation(self):
        """
         Test Cooperative Value (cVal) computation. An important step in computing Best Alternative (BA) value associated with edge on Graph of Utility.
        """
        gou_min_min_handle = SymbolicGraphOfUtlCooperativeGame(gou_handle=self.regret_synthesis_handle.graph_of_utls_handle,
                                                               ts_handle=self.regret_synthesis_handle.ts_handle,
                                                               dfa_handle=self.regret_synthesis_handle.dfa_handle,
                                                               ts_curr_vars=self.regret_synthesis_handle.ts_x_list,
                                                               dfa_curr_vars=self.regret_synthesis_handle.dfa_x_list,
                                                               sys_act_vars=self.regret_synthesis_handle.ts_robot_vars,
                                                               env_act_vars=self.regret_synthesis_handle.ts_human_vars,
                                                               ts_obs_vars=self.regret_synthesis_handle.ts_obs_list,
                                                               ts_utls_vars=self.regret_synthesis_handle.prod_utls_vars,
                                                               cudd_manager=self.cudd_manager,
                                                               monolithic_tr=MONOLITHIC_TR)

        self.cvals: ADD = gou_min_min_handle.solve(verbose=False, print_layers=False)

        self.assertEqual(gou_min_min_handle.init_state_value, 1, "Error computing cVal on the Graph of Utility constrcuted by explicitly rolling out the original graph.")

        # ensure winning strategy exists assuming human to be cooperative
        self.assertNotEqual(self.cvals, self.cudd_manager.addZero(), "Could not synthesize a winning strategy for cooperative game!!!")

        # ensure that you reach the fixed point correctly
        self.assertEqual(max(gou_min_min_handle.winning_states.keys()), 4, "Mismatch in # of iterations required to reach a fixed point for Min-Min game.")


    def test_5_graph_of_best_response_construction(self):
        """
         Test Graph of Best Response Construction given the Graph of Utiltiy and BA for every edge
        """
        gou_min_min_handle = SymbolicGraphOfUtlCooperativeGame(gou_handle=self.regret_synthesis_handle.graph_of_utls_handle,
                                                               ts_handle=self.regret_synthesis_handle.ts_handle,
                                                               dfa_handle=self.regret_synthesis_handle.dfa_handle,
                                                               ts_curr_vars=self.regret_synthesis_handle.ts_x_list,
                                                               dfa_curr_vars=self.regret_synthesis_handle.dfa_x_list,
                                                               sys_act_vars=self.regret_synthesis_handle.ts_robot_vars,
                                                               env_act_vars=self.regret_synthesis_handle.ts_human_vars,
                                                               ts_obs_vars=self.regret_synthesis_handle.ts_obs_list,
                                                               ts_utls_vars=self.regret_synthesis_handle.prod_utls_vars,
                                                               cudd_manager=self.cudd_manager,
                                                               monolithic_tr=MONOLITHIC_TR)

        cvals: ADD = gou_min_min_handle.solve(verbose=False, print_layers=False)

        # compute the best alternative from each edge for cumulative payoff
        self.regret_synthesis_handle.graph_of_utls_handle.get_best_alternatives(cooperative_vals=cvals,
                                                                                mod_act_dict=self.regret_synthesis_handle.mod_act_dict,
                                                                                print_layers=False,
                                                                                verbose=False)

        # construct additional boolean vars for set of best alternative values
        self.regret_synthesis_handle.prod_ba_vars: List[ADD] = self.regret_synthesis_handle._create_symbolic_lbl_vars(state_lbls=self.regret_synthesis_handle.graph_of_utls_handle.ba_set,
                                                                                                                      state_var_name='r',
                                                                                                                      add_flag=True)
        # construct of Best response G^{br}
        self.graph_of_br_handle = HybridGraphOfBR(curr_vars=self.regret_synthesis_handle.ts_x_list,
                                                  lbl_vars=self.regret_synthesis_handle.ts_obs_list,
                                                  robot_action_vars=self.regret_synthesis_handle.ts_robot_vars,
                                                  human_action_vars=self.regret_synthesis_handle.ts_human_vars,
                                                  task=self.regret_synthesis_handle.ts_handle.task,
                                                  domain=self.regret_synthesis_handle.ts_handle.domain,
                                                  ts_state_map=self.regret_synthesis_handle.ts_handle.pred_int_map,
                                                  ts_states=self.regret_synthesis_handle.ts_handle.ts_states,
                                                  manager=self.cudd_manager,
                                                  weight_dict=self.regret_synthesis_handle.ts_handle.weight_dict,
                                                  seg_actions=self.regret_synthesis_handle.ts_handle.actions,
                                                  ts_state_lbls=self.regret_synthesis_handle.ts_handle.state_lbls,
                                                  dfa_state_vars=self.regret_synthesis_handle.dfa_x_list,
                                                  sup_locs=self.regret_synthesis_handle.sup_locs,
                                                  top_locs=self.regret_synthesis_handle.top_locs,
                                                  ts_handle=self.regret_synthesis_handle.ts_handle,
                                                  dfa_handle=self.regret_synthesis_handle.dfa_handle,
                                                  symbolic_gou_handle=self.regret_synthesis_handle.graph_of_utls_handle,
                                                  prod_ba_vars=self.regret_synthesis_handle.prod_ba_vars)
        
        self.graph_of_br_handle.construct_graph_of_best_response(mod_act_dict=self.regret_synthesis_handle.mod_act_dict,
                                                                 print_layers=False,
                                                                 print_leaf_nodes=False,
                                                                 verbose=False,
                                                                 debug=True)
        
        # verify # of edges; # of states; # of leaf nodes required to constuct the graph.
        no_of_edges: int = self.graph_of_br_handle.ecount
        no_of_states: int = self.graph_of_br_handle.scount
        no_of_leaf_nodes: int = self.graph_of_br_handle.lcount

        leaf_values: int = self.graph_of_br_handle.leaf_vals
        fp_layer: int =  max(self.graph_of_br_handle.open_list.keys())
        
        self.assertEqual(no_of_edges, 30, "Error constructing Graph of Best Response. Mismatch in # of Edges")
        self.assertEqual(no_of_states, 12, "Error constructing Graph of Best Response. Mismatch in # of states")
        self.assertEqual(no_of_leaf_nodes, 4, "Error constructing Graph of Best Response. Mismatch in # of Leaf nodes")

        self.assertEqual(leaf_values, set([0, 1, 2]), "Error constructing Graph of Best Response. Mismatch in regret values associated with leaf nodes")
        self.assertEqual(fp_layer, 8, "Error constructing Graph of Best Response. Mismatch in # of layers required to construct the graph")

        self.reg_str_synthesis()



if __name__ == "__main__":
    unittest.main()
