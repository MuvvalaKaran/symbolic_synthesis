import re 
import sys

from functools import reduce
from typing import Union, List


from cudd import Cudd, BDD, ADD

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import SymbolicAddDFA, SymbolicWeightedTransitionSystem


class SymbolicDijkstraSearch(BaseSymbolicSearch):

    def __init__(self,
                 ts_handle: SymbolicWeightedTransitionSystem,
                 dfa_handle: SymbolicAddDFA,
                 ts_curr_vars: List[ADD],
                 ts_next_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 dfa_next_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = ts_handle.sym_add_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.ts_x_list = ts_curr_vars
        self.ts_y_list = ts_next_vars
        self.dfa_x_list = dfa_curr_vars
        self.dfa_y_list = dfa_next_vars
        self.ts_transition_fun_list = ts_handle.sym_tr_actions
        self.dfa_transition_fun = dfa_handle.dfa_bdd_tr
        self.ts_add_sym_to_curr_state_map: dict = ts_handle.predicate_add_sym_map_curr.inv
        self.ts_bdd_sym_to_curr_state_map: dict = ts_handle.predicate_sym_map_curr.inv
        self.ts_bdd_sym_to_S2obs_map: dict = ts_handle.predicate_sym_map_lbl.inv
        self.ts_add_sym_to_S2obs_map: dict = ts_handle.predicate_add_sym_map_lbl.inv
        self.dfa_bdd_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_sym_map_curr.inv
        self.dfa_add_sym_to_curr_state_map: dict = dfa_handle.dfa_predicate_add_sym_map_curr.inv
        self.obs_add = ts_handle.sym_add_state_labels
        self.tr_action_idx_map = ts_handle.tr_action_idx_map

        # create corresponding cubes to avoid repetition
        self.ts_xcube = reduce(lambda x, y: x & y, self.ts_x_list)
        self.ts_ycube = reduce(lambda x, y: x & y, self.ts_y_list)
        self.ts_obs_cube = reduce(lambda x, y: x & y, self.ts_obs_list)

        self.dfa_xcube = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.dfa_ycube = reduce(lambda x, y: x & y, self.dfa_y_list)

        # composed graph consists of state S, Z and hence are function TS and DFA vars
        self.prod_xlist = self.ts_x_list + self.dfa_x_list
        self.prod_ylist = self.ts_y_list + self.dfa_y_list
        self.prod_xcube = reduce(lambda x, y: x & y, self.prod_xlist)
        self.prod_ycube = reduce(lambda x, y: x & y, self.prod_ylist)


        # composed monolithic TR
        self.composed_tr_list = self._construct_composed_tr_function()
    

    def _construct_composed_tr_function(self) -> List[ADD]:
        """
        A function that construct that conjoints the TR function of the TS and DFA along with S2P (state to obs BDD).

        Note: We prime the S2P BDD because we want to extract the next state in the DFA after we evolve over the TS.
        """

        obs_bdd_prime = self.obs_add.swapVariables(self.ts_x_list, self.ts_y_list) 
        composed_tr_list = []
        for tr_action in self.ts_transition_fun_list:
            composed_tr = tr_action & obs_bdd_prime & self.dfa_transition_fun
            composed_tr_list.append(composed_tr)
        
        return composed_tr_list
    

    def _get_max_tr_action_cost(self) -> int:
        """
        A helper function that retireves the highest cost amongst all the transiton function costs
        """
        _max = 0
        for tr_action in self.ts_transition_fun_list:
                action_cost = tr_action.findMax()
                action_cost_int = int(re.findall(r'\d+', action_cost.__repr__())[0])
                if action_cost_int > _max:
                    _max = action_cost_int
        
        return _max
    

    def _construct_adj_mat(self, verbose: bool = False) -> ADD:
        """
        A helper function that create the Adjacency matric used to compute the image of the successor states.
        """
        
        obs_bdd_prime = self.obs_add.swapVariables(self.ts_x_list, self.ts_y_list) 
        composed_tr = self.manager.addZero()
        for tr_action in self.ts_transition_fun_list:
            composed_tr |= tr_action & self.dfa_transition_fun & obs_bdd_prime
        
        # We use ~T to obtain a 0-1 ADD (as required by ITE).
        composed_tr = (~composed_tr).ite(self.manager.plusInfinity(), composed_tr)
        if verbose:
            print("************************ Adjacency Matrix*******************")
            composed_tr.display(3)
        return composed_tr 
    
    
    def composed_symbolic_dijkstra_wLTL(self, verbose: bool = False):
        """
        A function that compose the TR function from the Transition system and DFA and search symbolically over the product graph.

        TODO: Modify the implementation where we retrieve the plan bu starting from the highest cost Action and them iterating 
        over them in a descending manner.  
        """
        open_list = {}
        closed = self.manager.addZero()

        c_max = self._get_max_tr_action_cost()
        empty_bucket_counter: int = 0
        g_val = self.manager.addZero()
        if g_val.isZero():
            g_layer = 0

        # add the init state to ite respective DFA state. Note, we could start in some other state than the usual T0_init
        open_list[g_layer] = self.init_TS & self.init_DFA

        while not self.target_DFA <= open_list[g_layer].existAbstract(self.ts_xcube):
            # remove all states that have been explored
            open_list[g_layer] = open_list[g_layer] & ~closed

            # If unexpanded states exist ... 
            if not open_list[g_layer].isZero():
                if verbose:
                    print(f"********************Layer: {g_layer }**************************")
                # reset the empty bucket counter 
                empty_bucket_counter = 0
                # Add states to be expanded next to already expanded states
                closed |= open_list[g_layer]

                for prod_tr_action in self.composed_tr_list:
                    # first get the corresponding transition action cost (constant at the terminal node)
                    action_cost = prod_tr_action.findMax()
                    step = g_val + action_cost
                    step_val = int(re.findall(r'\d+', step.__repr__())[0])

                    # compute the image of the TS states 
                    image_prod_add = self.image_per_action(trans_action=prod_tr_action,
                                                           From=open_list[g_layer],
                                                           xcube=self.prod_xcube,
                                                           x_list=self.prod_xlist,
                                                           y_list=self.prod_ylist)
                    
                    if image_prod_add.isZero():
                        continue

                    prod_image_restricted = image_prod_add.existAbstract(self.ts_obs_cube)
                
                    if verbose:
                        self.get_prod_states_from_dd(dd_func=image_prod_add, obs_flag=False)
                    
                    # if the bucket exists then take the union else initialize the bucket
                    if step_val in open_list:
                        open_list[step_val] |= prod_image_restricted
                    else:
                        open_list[step_val] = prod_image_restricted

            
            else:
                empty_bucket_counter += 1
                # If Cmax consecutive layers are empty. . .
                if empty_bucket_counter == c_max:
                    print("No plan exists! Terminating algorithm.")
                    sys.exit(-1)
            

            g_val = g_val + self.manager.addOne()
            g_layer += 1

            # keep updating g_layer up until the most recent bucket
            while g_layer not in open_list:
                g_val = g_val + self.manager.addOne()
                g_layer += 1
        
        
        open_list[g_layer] = open_list[g_layer] & self.target_DFA

        if verbose:
            print("********************The goal state encountered is***********************")
            self.get_prod_states_from_dd(dd_func=open_list[g_layer], obs_flag=False)
        
        print(f"Found a plan with least cost lenght {g_layer}, Now retireving it!")

        return self.retrieve_composed_dijkstra(max_layer=g_layer, freach_list=open_list, verbose=verbose)
    

    def retrieve_composed_dijkstra(self, max_layer: int, freach_list: dict, verbose: bool = False):
        """
        Retrieve the plan through Backward search by starting from the Goal state and computing the interseaction of Forwards and Backwards
         Reachable set. 
        """
        g_layer = self.manager.addConst(int(max_layer))
        g_int = int(re.findall(r'-?\d+', g_layer.__repr__())[0])
        print("Working Retrieval plan now")

        current_prod = freach_list[g_int]
        composed_prod_state = self.init_TS & self.init_DFA

        parent_plan = {}

        while not composed_prod_state <= freach_list[g_int]:
            new_current_prod = self.manager.addZero()
            for tr_num, prod_tr_action in enumerate(self.composed_tr_list):
                pred_prod= self.pre_per_action(trans_action=prod_tr_action,
                                               From=current_prod,
                                               ycube=self.prod_ycube,
                                               x_list=self.prod_xlist,
                                               y_list=self.prod_ylist)
                
                if pred_prod.isZero():
                    continue

                # first get the corresponding transition action cost (constant at the terminal node)
                action_cost_cnst = prod_tr_action.findMax()
                step = g_layer - action_cost_cnst
                if step.isZero():
                    step_val = 0
                else:
                    step_val = int(re.findall(r'-?\d+', step.__repr__())[0])
                    # there can be cases where step_val cam go negtive, we skip such iterations
                    if step_val < 0:
                        continue

                if pred_prod & freach_list[step_val] != self.manager.addZero():
                    # store the predecessor per action
                    tmp_current_prod = pred_prod & freach_list[step_val]

                    tmp_current_prod_res = (pred_prod & freach_list[step_val]).existAbstract(self.ts_obs_cube)
                    
                    self._append_dict_value_composed(parent_plan,
                                                     key_prod=tmp_current_prod_res,
                                                     action=self.tr_action_idx_map.inv[tr_num])
                    
                    new_current_prod |= tmp_current_prod_res 

                    g_layer = step
            
            current_prod = new_current_prod

            if g_layer.isZero():
                g_int = 0
            else:
                g_int = int(re.findall(r'-?\d+', g_layer.__repr__())[0])
            assert  g_int >= 0, "Error Retrieving a plan. FIX THIS!!"

            if verbose:
                print(f"********************Layer: {g_int}**************************")
                self.get_prod_states_from_dd(dd_func=tmp_current_prod, obs_flag=False)
            
        return parent_plan
    

    def ADD_existAbstract(self, dd_func: ADD) -> ADD:
        """
        In the ADD implementation of Dijkstra's algorithm, we compute the image anbd shortest 1-step distance by taking Semi Ring matrix operation as defined 
        by Fabio Somenzi in his ADD paper.

        As all the valid paths have value >= 0 and invalid paths have values plusInfinity, when we apply existAbstract := Apply(+, func|x, func|~x),
        The invalid edges with plusInfinity add to finite int weights and end up giving infinity. 

        Hence, to existAbstract(), we first conver the ADD to 0-1 ADD and then do the existential Abstraction 
        """

        # generate cubes, 
        int_leaves = [i[1] for i in  list(dd_func.generate_cubes())] 
    

        # create a set to eliminate duplicates
        int_cubes = set(int_leaves)

        # loop over each int leaf, extract the bdd, then perform existential Abstraction and ocnvert back to ADD
        for num in int_cubes:
            dd: BDD = dd_func.bddInterval(num, num)
            new_dd: ADD = dd.existAbstract(self.ts_obs_cube.bddPattern()).toADD()

            # now add the int values at the leaves back to the ADD
            new_dd = (~new_dd).ite(self.manager.plusInfinity(), self.manager.addConst(num))
        
        assert new_dd <= dd_func, "Error Computing the existential Abstraction of successor states during forward search. FIX THIS!!!!"
        # TODO: need to append all the dds and then return 
        return new_dd


    
    def ADD_composed_symbolic_dijkstra_wLTL(self, verbose: bool = False, debug: bool = False):
        """
        This function implemented Dijstra's symbolic composed algorithm using ADDs.

        This algorithm is based off of ADDA* algorithm proposed by  Hansen, Zhou and Feng in 2002-
         "Symbolic heuristic search using decision diagrams."
        
        Using this approach - we can use
        1. Single ADD instead of buckets (vector of) of BDD.
        2. Keeping Track of G values as part of the staates instead of extracting them and then storing their corresponding BDDs.

        Boolean Operators on ADDs are only defined when the leaves of ADD are 0 & 1. 

        bddThreshold(value: int) - return the BDD associated with leaves that are >= value
        bddStrictThreshold(value: int) - return the BDD associated with leaves that are (stictly) > value
        bddInterval(value: int) - return the BDD associated with leaves that satisfy lower <= value <= upper

        Use isConstant() and isNonConstant() to check if it is Constant ADD or not.
        """
        self.old_code()
        # self.manager.setBackground(self.manager.plusInfinity())

        # # create TS adjacency map
        # ts_adj_mat = reduce(lambda x, y: x | y, self.ts_transition_fun_list)
        # ts_adj_mat = (~ts_adj_mat).ite(self.manager.plusInfinity(), ts_adj_mat)

        # # create dfa adjaceny map
        # dfa_adj_mat = (~self.dfa_transition_fun).ite(self.manager.plusInfinity(), self.manager.addZero())

        # # create Obd adjacency map
        # obs_add = self.obs_add.ite(self.manager.addZero(), self.manager.plusInfinity())


        # # get the initial state, evolve over the TS
        # init_TS = self.init_TS.ite(self.manager.addZero(), self.manager.plusInfinity())
        # init_dfa = self.init_DFA.ite(self.manager.addZero(), self.manager.plusInfinity())
        # init_state_obs = obs_add + init_TS

        # target_dfa = self.target_DFA.ite(self.manager.addZero(), self.manager.plusInfinity())

        # open_add = init_TS + init_dfa + init_state_obs

        # while not open_add.isZero():
        #     # next transition states
            
        #     nxt_ts_state = ts_adj_mat.triangle(open_add, self.ts_x_list).swapVariables(self.ts_x_list, self.ts_y_list)

        #     # get the observation
        #     state_obs = obs_add.restrict(nxt_ts_state)

        #     # get the next DFA states          
        #     nxt_dfa_states = dfa_adj_mat.triangle(nxt_ts_state + state_obs + init_dfa, self.dfa_x_list).swapVariables(self.dfa_x_list, self.dfa_y_list)

        #     open_add = nxt_dfa_states
        #     nxt_ts_state = nxt_dfa_states
        #     # get the prod states,
        #     print("NXT DFA STate")

        
        
        
        
        
        
        
    
    def old_code(self, verbose: bool = False):
        ####### OLD CODE #####################
        self.manager.setBackground(self.manager.plusInfinity())
        adj_mat: ADD = self._construct_adj_mat(verbose=False)
        init_TS = self.init_TS.ite(self.manager.addZero(), self.manager.plusInfinity())
        init_dfa = self.init_DFA.ite(self.manager.addZero(), self.manager.plusInfinity())

        composed_init = (self.init_TS & self.init_DFA).ite(self.manager.addZero(), self.manager.plusInfinity())
        open_add = composed_init
        # closed = self.manager.addZero()
        
        # if debug: 
        closed_add = self.manager.plusInfinity()
        # closed_add = self.manager.bddZero()
        

        while not open_add.isZero():
            # open_add = closed_add - open_add

            # finding the states with minimum g value
            new_add = open_add.min(open_add)
            # closed_add |= new_add.bddPattern()
            new_closed_add = closed_add.min(new_add)

            if new_closed_add.agreement(closed_add).isOne():
                print("Reached a Fixed Point, No path exisits.")
            
            closed_add = new_closed_add


            # f_min = open_add.findMin()
            # # get the add associated with it
            # if f_min.isZero():
            #     f_min_int = 0
            #     # get the BDD associated with the min f value
            #     new_bdd: BDD = open_add.bddInterval(f_min_int, f_min_int)
            #     new_add: ADD = new_bdd.toADD()

            #     # after converting ADD, we need to reset the false edges (labelled as 0 due to BDDtoADD comversion) to +infinity
            #     new_add: ADD = new_add.ite(f_min, self.manager.plusInfinity())

            # else:
            #     f_min_int = int(re.findall(r'-?\d+', f_min.__repr__())[0])
            #     new = open_add.bddInterval(f_min_int, f_min_int).toADD()

            # open_add = open_add - new_add   # FIX THIS!!
            # Keep track of the lowest g associared with each state
            # closed_add = closed_add.min(open_add) 
            if not new_add.restrict(self.target_DFA) == self.manager.plusInfinity():
                return 
            # if new_add.restrict(self.target_DFA) == self.manager.plusInfinity():
            # closed_add = closed_add.min(open_add)
            nxt: ADD = adj_mat.triangle(new_add, self.prod_xlist).swapVariables(self.prod_xlist, self.prod_ylist)

            nxt = self.ADD_existAbstract(dd_func=nxt)
            # nxt_min = nxt.findMin()
            # nxt_min_int = int(re.findall(r'-?\d+', nxt_min.__repr__())[0])

            # nxt_max = nxt.findMax()
            # nxt_max_int = int(re.findall(r'-?\d+', nxt_max.__repr__())[0])

            # assert nxt_max_int >= nxt_min_int, "Error while converting successors to current state on the composed graph using existAbstract"

            # get the BDD associated with the min f value
            # + 1 can be replaced with 
            # nxt_bdd: BDD = nxt.bddInterval(nxt_min_int, nxt_min_int).existAbstract(self.ts_obs_cube.bddPattern())
            # nxt_add: ADD = nxt_bdd.toADD().ite()

            if verbose:
                self.get_prod_states_from_dd(dd_func=nxt, obs_flag=False)

            # retain the minimum onces only
            # open_add: ADD = new_add.min(nxt)
            open_add: ADD = nxt

    
                



                
















