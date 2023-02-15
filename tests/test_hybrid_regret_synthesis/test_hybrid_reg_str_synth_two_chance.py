'''
 This files tests all steps executed during synthesis of regret-minimizing strategies.

  1. We first test original graph construction 
  2. We then test min-max value iteration on the original graph - the min-max value determines the minimum energy budget for the regret graphs
  3. Then, we test Graph of Utiity construction
  4. We test cVal computation on the graph of Utility, an intermediate computation need to compute Best Alternate value associated with edge on the Graph of Utility. 
  5. Finally, we test Graph of Best Response construction and
  4. We finally compute regret minimizing strategies by playing min-max game on the Graph of Best Response. 
  5. We check if the strategies are correct, i.e., if the optimal strategy is give no, one, two ... chances to cooperate
     by comparing the synthesized optimal strategy with a the correct strategy. 
'''


import os
import unittest

from typing import List
from cudd import Cudd, ADD

# from src.algorithms.strategy_synthesis import AdversarialGame
from src.symbolic_graphs.strategy_synthesis_scripts import FrankaRegretSynthesis

from src.symbolic_graphs.hybrid_regret_graphs import HybridGraphOfBR

from src.algorithms.strategy_synthesis import GraphOfUtlCooperativeGame, GraphofBRAdvGame

# config flags 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DYNAMIC_VAR_ORDERING: bool = False

USE_LTLF: bool = True # Construct DFA from LTLf

SUP_LOC = []
TOP_LOC = []    


class TestRegretStrSynth(unittest.TestCase):
    """
     In this method we increment the robot's cost of operating in Robot region from 2 to 3. This allows the robot to give one additional
      opportunity for the human to be cooperative. 

     If we keep incrementing the cost by 1, the robot will 1 more or additional chance for the human to be collaboratively.
      We verify this behavior in Test_5
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.formulas = ['F(p01) | F(p17)']

        domain_file_path: str = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path: str = PROJECT_ROOT + "/pddl_files/problem.pddl"

        wgt_dict = {
                "transit" : 1,
                "grasp"   : 1,
                "transfer": 1,
                "release" : 1,
                "human": 0
                }

        cls.cudd_manager = Cudd()

        cls.regret_synthesis_handle = FrankaRegretSynthesis(domain_file=domain_file_path,
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
                                                        plot_obs=False,
                                                        plot=False)
        
        cls.regret_synthesis_handle.build_abstraction()


    def test_1_original_graph_construction(self):
        """
         Test PDDL to Two-player Game construction with edge weights
        """
        total_vars: int = self.cudd_manager.size()
            
        self.assertEqual(total_vars,
                         18,
                         msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Weighted Abstraction for formula {self.formulas[0]}")
        
        self.assertEqual(self.regret_synthesis_handle.ts_handle.ecount,
                         898,
                         msg=f"Mismatch in the # of edges in the Symbolic Weighted Abstraction for formula {self.formulas[0]}.")
        
        

    def test_2_min_max_value_iteration(self):
        """
         Test adversarial game playing for determining minimum energy budget 
        """
        # stop after computing the Min-max value.
        with self.assertRaises(SystemExit) as cm:
            self.regret_synthesis_handle.solve(verbose=False, just_adv_game=True, run_monitor=True)

            # verify min max value
            self.assertEqual(TestRegretStrSynth.regret_synthesis_handle.min_energy_budget, 12, "Error computing aVal on the original Two-player game") 

            # verify regret budget
            self.assertEqual(TestRegretStrSynth.regret_synthesis_handle.reg_energy_budget, 15, "Error computing Regret Budget for Strategy synthesis.")
        
        self.assertEqual(cm.exception.code, -1)


    def test_3_graph_of_utility_constrcution(self):
        """
         Test Graph of Utility (unrolling original graph) Construction
        """
        self.regret_synthesis_handle.build_add_graph_of_utility(verbose=False, just_adv_game=False)

        # verify # of edges; # of states; # of leaf nodes required to constuct the graph.
        no_of_edges: int = self.regret_synthesis_handle.graph_of_utls_handle.ecount
        no_of_states: int = self.regret_synthesis_handle.graph_of_utls_handle.scount
        no_of_leaf_nodes: int = self.regret_synthesis_handle.graph_of_utls_handle.lcount

        leaf_values: int = self.regret_synthesis_handle.graph_of_utls_handle.leaf_vals
        fp_layer: int =  max(self.regret_synthesis_handle.graph_of_utls_handle.open_list.keys())
        
        self.assertEqual(no_of_edges, 2623, "Error constructing Graph of Utility. Mismatch in # of Edges")
        self.assertEqual(no_of_states, 483, "Error constructing Graph of Utility. Mismatch in # of states")
        self.assertEqual(no_of_leaf_nodes, 167, "Error constructing Graph of Utility. Mismatch in # of Leaf nodes")

        self.assertEqual(leaf_values, set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]), "Error constructing Graph of Utility. Mismatch in # utility values associated with leaf nodes")
        self.assertEqual(fp_layer, 18, "Error constructing Graph of Utility. Mismatch in # of layers required to construct the graph")
        

    def test_4_cVal_computation(self):
        """
         Test Cooperative Value (cVal) computation. An important step in computing Best Alternative (BA) value associated with edge on Graph of Utility.
        """
        gou_min_min_handle = GraphOfUtlCooperativeGame(prod_handle=self.regret_synthesis_handle.graph_of_utls_handle,
                                                       ts_handle=self.regret_synthesis_handle.ts_handle,
                                                       dfa_handle=self.regret_synthesis_handle.dfa_handle,
                                                       ts_curr_vars=self.regret_synthesis_handle.ts_x_list,
                                                       dfa_curr_vars=self.regret_synthesis_handle.dfa_x_list,
                                                       sys_act_vars=self.regret_synthesis_handle.ts_robot_vars,
                                                       env_act_vars=self.regret_synthesis_handle.ts_human_vars,
                                                       ts_obs_vars=self.regret_synthesis_handle.ts_obs_list,
                                                       ts_utls_vars=self.regret_synthesis_handle.prod_utls_vars,
                                                       cudd_manager=self.cudd_manager)

        self.cvals: ADD = gou_min_min_handle.solve(verbose=False)

        self.assertEqual(gou_min_min_handle.init_state_value, 1, "Error computing cVal on the Graph of Utility constrcuted by explicitly rolling out the original graph.")

        # ensure winning strategy exists assuming human to be cooperative
        self.assertNotEqual(self.cvals, self.cudd_manager.addZero(), "Could not synthesize a winning strategy for cooperative game!!!")

        # ensure that you reach the fixed point correctly
        self.assertEqual(max(gou_min_min_handle.winning_states.keys()), 4, "Mismatch in # of iterations required to reach a fixed point for Min-Min game.")


    def test_5_graph_of_best_response_construction(self):
        """
         Test Graph of Best Response Construction given the Graph of Utiltiy and BA for every edge
        """
        gou_min_min_handle = GraphOfUtlCooperativeGame(prod_handle=self.regret_synthesis_handle.graph_of_utls_handle,
                                                       ts_handle=self.regret_synthesis_handle.ts_handle,
                                                       dfa_handle=self.regret_synthesis_handle.dfa_handle,
                                                       ts_curr_vars=self.regret_synthesis_handle.ts_x_list,
                                                       dfa_curr_vars=self.regret_synthesis_handle.dfa_x_list,
                                                       sys_act_vars=self.regret_synthesis_handle.ts_robot_vars,
                                                       env_act_vars=self.regret_synthesis_handle.ts_human_vars,
                                                       ts_obs_vars=self.regret_synthesis_handle.ts_obs_list,
                                                       ts_utls_vars=self.regret_synthesis_handle.prod_utls_vars,
                                                       cudd_manager=self.cudd_manager)

        cvals: ADD = gou_min_min_handle.solve(verbose=False)

        # compute the best alternative from each edge for cumulative payoff
        self.regret_synthesis_handle.graph_of_utls_handle.get_best_alternatives(cooperative_vals=cvals,
                                                                                mod_act_dict=self.regret_synthesis_handle.mod_act_dict,
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
                                               prod_ba_vars=self.regret_synthesis_handle.prod_ba_vars, 
                                               prod_succ_ba_vars=None)
        
        self.graph_of_br_handle.construct_graph_of_best_response(mod_act_dict=self.regret_synthesis_handle.mod_act_dict,
                                                                 print_leaf_nodes=False,
                                                                 verbose=False,
                                                                 debug=True)
        
        # verify # of edges; # of states; # of leaf nodes required to constuct the graph.
        no_of_edges: int = self.graph_of_br_handle.ecount
        no_of_states: int = self.graph_of_br_handle.scount
        no_of_leaf_nodes: int = self.graph_of_br_handle.lcount

        leaf_values: int = self.graph_of_br_handle.leaf_vals
        fp_layer: int =  max(self.graph_of_br_handle.open_list.keys())
        
        self.assertEqual(no_of_edges, 5625, "Error constructing Graph of Best Response. Mismatch in # of Edges")
        self.assertEqual(no_of_states, 1028, "Error constructing Graph of Best Response. Mismatch in # of states")
        self.assertEqual(no_of_leaf_nodes, 368, "Error constructing Graph of Best Response. Mismatch in # of Leaf nodes")

        self.assertEqual(leaf_values, set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), "Error constructing Graph of Best Response. Mismatch in regret values associated with leaf nodes")
        self.assertEqual(fp_layer, 18, "Error constructing Graph of Best Response. Mismatch in # of layers required to construct the graph")

        self.reg_str_synthesis()
        

    def reg_str_synthesis(self):
        """
         Tests Regret-minimizing strategy synthesis by playing Min-MAx game on the Graph of Best Response. 

         Note: This method only check the correctness of the code. The validity of the synthesized strategy is done by rolling out the strategy. 
        """
        # compute regret-minmizing strategies
        self.gbr_min_max_handle =  GraphofBRAdvGame(prod_gbr_handle=self.graph_of_br_handle,
                                                    prod_gou_handle=self.regret_synthesis_handle.graph_of_utls_handle,
                                                    ts_handle=self.regret_synthesis_handle.ts_handle,
                                                    dfa_handle=self.regret_synthesis_handle.dfa_handle,
                                                    ts_curr_vars=self.regret_synthesis_handle.ts_x_list,
                                                    dfa_curr_vars=self.regret_synthesis_handle.dfa_x_list,
                                                    ts_obs_vars=self.regret_synthesis_handle.ts_obs_list,
                                                    prod_utls_vars=self.regret_synthesis_handle.prod_utls_vars,
                                                    prod_ba_vars=self.regret_synthesis_handle.prod_ba_vars,
                                                    sys_act_vars=self.regret_synthesis_handle.ts_robot_vars,
                                                    env_act_vars=self.regret_synthesis_handle.ts_human_vars,
                                                    cudd_manager=self.cudd_manager)

        # calling the solve() method
        self.reg_str: ADD = self.gbr_min_max_handle.solve(verbose=False)

        self.assertEqual(self.gbr_min_max_handle.init_state_value, 11, "Error computing optimal regret value on the Graph of Best Response.")

        # ensure regret minimizing strategy exists
        self.assertNotEqual(self.reg_str, self.cudd_manager.addZero(), "Could not synthesize a Regret Minimizing strategy!!!")

        # ensure that you reach the fixed point correctly
        self.assertEqual(max(self.gbr_min_max_handle.winning_states.keys()), 5, "Mismatch in # of iterations required to reach a fixed point for Min-Min game.")

        self.max_layer: int = max(self.gbr_min_max_handle.winning_states.keys())

        self.reg_str_optimality()


    def reg_str_optimality(self):
        """
         Test if the optimal behaviors are in-line with the expected behaviors. 
        """
        # roll out strategy and check verify optimal strategy
        expected_strategy = ['transit b1', 'transit b1']

        # robot should human once chance before becoming conservative
        # STEP 1
        act_cube, _ = self.gbr_min_max_handle.get_strategy_and_val(strategy=self.reg_str,
                                                                    max_layer=self.max_layer,
                                                                    curr_prod_state=self.gbr_min_max_handle.init_prod,
                                                                    curr_prod_tuple=None,
                                                                    print_all_act_vals=False)
        
        self.assertNotEqual(self.gbr_min_max_handle.ts_sym_to_robot_act_map.inv[expected_strategy[0]] & act_cube,
                            self.cudd_manager.addZero(),
                            "Did not synthesize the correct regret minimizing behavior!!!" )
        
        sym_lbl_cubes = self.gbr_min_max_handle._create_lbl_cubes()

        # get the successor state
        curr_ts_state: ADD = self.gbr_min_max_handle.init_prod.existAbstract(self.gbr_min_max_handle.dfa_xcube & self.gbr_min_max_handle.sys_env_cube & self.gbr_min_max_handle.prod_utls_cube & self.gbr_min_max_handle.prod_ba_cube).bddPattern().toADD()   # to get 0-1 ADD
        curr_dfa_state: ADD = self.gbr_min_max_handle.init_prod.existAbstract(self.gbr_min_max_handle.ts_xcube & self.gbr_min_max_handle.ts_obs_cube & self.gbr_min_max_handle.sys_env_cube & self.gbr_min_max_handle.prod_utls_cube & self.gbr_min_max_handle.prod_ba_cube ).bddPattern().toADD()
        curr_dfa_tuple: int = self.gbr_min_max_handle.dfa_sym_to_curr_state_map[curr_dfa_state]
        curr_ts_tuple: tuple = self.gbr_min_max_handle.ts_handle.get_state_tuple_from_sym_state(sym_state=curr_ts_state, sym_lbl_xcube_list=sym_lbl_cubes)
        curr_prod_utl: int = self.gbr_min_max_handle.prod_gou_handle.predicate_sym_map_utls.inv[self.gbr_min_max_handle.init_prod.existAbstract(self.gbr_min_max_handle.ts_xcube & self.gbr_min_max_handle.ts_obs_cube & self.gbr_min_max_handle.dfa_xcube & self.gbr_min_max_handle.prod_ba_cube)]
        curr_prod_ba: int = self.gbr_min_max_handle.prod_gbr_handle.predicate_sym_map_ba.inv[self.gbr_min_max_handle.init_prod.existAbstract(self.gbr_min_max_handle.ts_xcube & self.gbr_min_max_handle.ts_obs_cube & self.gbr_min_max_handle.dfa_xcube & self.gbr_min_max_handle.prod_utls_cube)]
        
        # STEP 2
        # look up the next tuple 
        next_prod_tuple = self.gbr_min_max_handle.prod_gbr_handle.prod_adj_map[(curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba)]['transit b1']['human-move b1 l8']
        next_prod_sym = self.gbr_min_max_handle.get_sym_prod_state_from_tuple(next_prod_tuple)

        act_cube, _ = self.gbr_min_max_handle.get_strategy_and_val(strategy=self.reg_str,
                                                                    max_layer=self.max_layer,
                                                                    curr_prod_state=next_prod_sym,
                                                                    curr_prod_tuple=None,
                                                                    print_all_act_vals=False)
        
        self.assertNotEqual(self.gbr_min_max_handle.ts_sym_to_robot_act_map.inv[expected_strategy[1]] & act_cube,
                            self.cudd_manager.addZero(),
                            "Did not synthesize the correct regret minimizing behavior!!!" )

        # this has to be done to ensure that
        # 1) the strategy synthesized does indeed reach an accepting state (complete the task), and
        # 2) to ensure that the code does not seg fault.
        self.gbr_min_max_handle.roll_out_strategy(strategy=self.reg_str,
                                                  ask_usr_input=False,
                                                  verbose=False)



if __name__ == "__main__":
    unittest.main()
