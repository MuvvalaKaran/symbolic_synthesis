import re 
import sys

from functools import reduce
from typing import List


from cudd import Cudd, BDD, ADD
from utls import deprecated
from yaml import warnings


from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import SymbolicAddDFA, SymbolicWeightedTransitionSystem


class SymbolicDijkstraSearch(BaseSymbolicSearch):
    """
    Given a Transition systenm, and a DFA associated with one Formula, this class computes the minimum cost path
    by searching over the composed graph using the Symbolic Dijkstras algorithm.

    Algorithm from Peter Kissmann's PhD thesis - Symbolic Search in Planning and General Game Playing.
     Link - https://media.suub.uni-bremen.de/handle/elib/405
    """
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

        # composed graph consists of state S, Z and hence are function of TS and DFA vars
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
            if not tr_action.isZero():
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
                    if prod_tr_action.isZero():
                        continue
                    
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
                    prod_image_restricted = prod_image_restricted.bddPattern().toADD()
                
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

                if pred_prod & freach_list.get(step_val, self.manager.addZero()) != self.manager.addZero():
                    # store the predecessor per action
                    tmp_current_prod = pred_prod & freach_list[step_val]

                    tmp_current_prod_res = (pred_prod & freach_list[step_val]).existAbstract(self.ts_obs_cube)
                    
                    self._append_dict_value_composed(parent_plan,
                                                     key_prod=tmp_current_prod_res,
                                                     action=self.tr_action_idx_map.inv[tr_num])
                    
                    current_prod = tmp_current_prod_res 

                    g_layer = step
                    break

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


    
    def ADD_composed_symbolic_dijkstra_wLTL(self, verbose: bool = False):
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
        warnings.warn("This code is still a Work In Progress. It does not work for non-uniform weights and retrieving plans.")

        self.manager.setBackground(self.manager.plusInfinity())
        adj_mat: ADD = self._construct_adj_mat(verbose=False)
        init_TS = self.init_TS.ite(self.manager.addZero(), self.manager.plusInfinity())
        init_dfa = self.init_DFA.ite(self.manager.addZero(), self.manager.plusInfinity())

        composed_init = (self.init_TS & self.init_DFA).ite(self.manager.addZero(), self.manager.plusInfinity())
        open_add = composed_init
        # closed = self.manager.addZero()
        closed_add = self.manager.plusInfinity()
        layer = 0
        

        while not open_add.isZero():
            # open_add = closed_add - open_add
            # finding the states with minimum g value
            new_add = open_add.min(open_add)
            new_closed_add = closed_add.min(new_add)

            if new_closed_add.agreement(closed_add).isOne():
                print("Reached a Fixed Point, No path exisits.")
            
            closed_add = new_closed_add

            # open_add = open_add - new_add   # FIX THIS!!
            # Keep track of the lowest g associared with each state
            # closed_add = closed_add.min(open_add) 
            if not new_add.restrict(self.target_DFA) == self.manager.plusInfinity():
                print(f"*****************The Shortest length path is {layer}****************************")
                return
                return self.retrieve_ADD_composed_symbolic_dijkstra_wLTL(freach=closed_add, verbose=verbose, maxd =layer)
           
            # closed_add = closed_add.min(open_add)
            nxt: ADD = adj_mat.triangle(new_add, self.prod_xlist).swapVariables(self.prod_xlist, self.prod_ylist)

            nxt = self.ADD_existAbstract(dd_func=nxt)

            if verbose:
                # self.get_prod_states_from_dd(dd_func=nxt, obs_flag=False)
                self.add_get_prod_stated_from_dd(dd_func=nxt, obs_flag=False)

            # retain the minimum onces only
            # open_add: ADD = new_add.min(nxt)
            open_add: ADD = nxt
            layer += 1
    
    @deprecated
    def retrieve_ADD_composed_symbolic_dijkstra_wLTL(self, freach: ADD, maxd: int, verbose: bool = False):
        """
        Backward search algorithm to retreive the action to be taken at each step
        """

        adj_mat: ADD = self._construct_adj_mat(verbose=False)

        # revesr the edges for backward search 
        adj_mat = adj_mat.swapVariables(self.prod_xlist, self.prod_ylist)

        test_target = (~self.target_DFA).ite(self.manager.plusInfinity(), self.manager.addConst(maxd))

        # start from the accepting state and do a backards search
        composed_acc = freach.agreement(test_target)

        raise NotImplementedError()



    
                



                
















