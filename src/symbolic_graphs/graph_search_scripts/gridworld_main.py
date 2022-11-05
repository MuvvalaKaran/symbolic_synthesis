'''
All functions relevant to construction of the symbolic gridworld Abstraction (Transition system).
'''
import sys
import time
import math
import warnings

from typing import Tuple, List, Dict, Union
from cudd import Cudd, BDD, ADD
from itertools import product


from src.explicit_graphs import CausalGraph


from src.symbolic_graphs import SymbolicDFA, SymbolicAddDFA
from src.symbolic_graphs import SymbolicTransitionSystem, SymbolicWeightedTransitionSystem

from src.algorithms.blind_search import SymbolicSearch, MultipleFormulaBFS
from src.algorithms.weighted_search import SymbolicDijkstraSearch, MultipleFormulaDijkstra
from src.algorithms.weighted_search import SymbolicBDDAStar, MultipleFormulaBDDAstar

from src.simulate_strategy import create_gridworld, \
     convert_action_dict_to_gridworld_strategy, plot_policy, convert_action_dict_to_gridworld_strategy_nLTL


from .base_main import BaseSymMain


class SimpleGridWorld(BaseSymMain):

    def __init__(self,
                 domain_file: str, 
                 problem_file: str,
                 formulas: Union[List, str],
                 manager: Cudd,
                 algorithm: str,
                 weight_dict: dict = {},
                 ltlf_flag: bool = True,
                 dyn_var_ord: bool = False,
                 verbose: bool = False,
                 plot_ts: bool = False,
                 plot_obs: bool = False,
                 plot_dfa: bool = False,
                 plot: bool = False,
                 create_lbls: bool = True):
        super().__init__(domain_file, problem_file, formulas, manager, plot_dfa, ltlf_flag, dyn_var_ord)

        self.algorithm: str = algorithm
        self.weight_dict: Dict[str, int] = weight_dict

        self.verbose: bool = verbose
        self.plot: bool = plot
        self.plot_ts: bool = plot_ts
        self.plot_obs: bool = plot_obs
        self.dyn_var_ordering: bool = dyn_var_ord

        self.create_lbls: bool = create_lbls


    def set_variable_reordering(self, make_tree_node: bool = False, **kwargs):
        """
        This function is called when DYNAMIC_VAR_ORDERING is True.

        Different ways to speed up the process
        1. AutodynaEnable() - Enable Dyanmic variable reordering
        2. ReorderingStatus() - Return the current reordering status and default method
        3. EnablingOrderingMonitoring() - Enable monitoring of a variable order 
        4. maxReorderings() - Read and set maximum number of variable reorderings 
        5. EnablereorderingReport() - Enable reporting of variable reordering

        MakeTreeNode() - Allows us to specify constraints over groups of variables. For example, we can constrain x, x'
        to always be contiguous. Thus, the relative ordering within the group is left unchanged. 

        MTR takes in two args -
        low: 
        size: 2 (grouping curr state vars and their corresponding primes together)

        """
        self.manager.autodynEnable()

        if make_tree_node:
            # Current, we follow the convention where we first build the TS variables, then the observations,
            # and finally the dfa variables. Within the TS and DFA, we pait vars and their primes together.
            # The observation variables are all grouped together as one.
            var_reorder_counter = 0  
            for i in range(self.manager.size()):
            # for i in range(kwargs['ts_sym_var_len']):
                if i<= kwargs['ts_sym_var_len']:
                    self.manager.makeTreeNode(2*i, 2)
                # elif i > kwargs['ts_sym_var_len'] and i <= kwargs['ts_obs_var_len']:
                #     manager.makeTreeNode(kwargs['ts_obs_var_len']*i, kwargs['ts_obs_var_len'])
                # elif i> kwargs['ts_obs_var_len'] + kwargs['ts_sym_var_len']:
                #     manager.makeTreeNode(2*i, 2)

        if self.verbose:
            self.manager.enableOrderingMonitoring()
        else:
            self.manager.enableReorderingReporting() 

    
    def build_abstraction(self):
        if self.algorithm in ['dijkstras','astar']:
            
            if len(self.weight_dict.keys()) == 0:
                warnings.warn("Please enter the weights associated with gridworld transitions. The actions for Gridworld are 'moveleft', 'moveright', 'moveup', 'movedown'")
                sys.exit(-1)

            # All vars (TS, DFA and Predicate) are of type ADDs
            sym_tr, ts_curr_state, ts_next_state, ts_lbl_states = self.build_weighted_add_abstraction()
            
            # The tuple contains the DFA handle, DFA curr and next vars in this specific order
            dfa_tr, dfa_curr_state, dfa_next_state = self.build_add_symbolic_dfa(sym_tr_handle=sym_tr)

        elif self.algorithm == 'bfs':
            sym_tr, ts_curr_state, ts_next_state, ts_lbl_states = self.build_bdd_abstraction()

            dfa_tr, dfa_curr_state, dfa_next_state = self.build_bdd_symbolic_dfa(sym_tr_handle=sym_tr)

        else:
            warnings.warn("Please enter a valid graph search algorthim. Currently Available - bfs (BDD), dijkstras (BDD/ADD), astar (BDD/ADD)")
        

        self.ts_handle: Union[SymbolicTransitionSystem, SymbolicWeightedTransitionSystem] = sym_tr
        self.dfa_handle_list: Union[SymbolicDFA, SymbolicAddDFA] = dfa_tr

        self.ts_x_list: List[BDD] = ts_curr_state
        self.ts_y_list: List[BDD] = ts_next_state
        self.ts_obs_list: List[BDD] = ts_lbl_states

        self.dfa_x_list: List[BDD] = dfa_curr_state
        self.dfa_y_list: List[BDD] = dfa_next_state


        
        if self.dyn_var_ordering:
            self.set_variable_reordering(make_tree_node=True,
                                         ts_sym_var_len=len(ts_curr_state),
                                         ts_obs_var_len=len(ts_lbl_states))
    
    def solve(self, verbose: bool = False) -> dict:
        """
        A function that calls the appropriate solver based on the algorithm specified and if single LTL of multiple formulas have been passed.
        """
        if len(self.formulas) > 1:
            start: float = time.time()
            if self.algorithm == 'dijkstras':
                graph_search = MultipleFormulaDijkstra(ts_handle=self.ts_handle,
                                                       dfa_handles=self.dfa_handle_list,
                                                       ts_curr_vars=self.ts_x_list,
                                                       ts_next_vars=self.ts_y_list,
                                                       dfa_curr_vars=self.dfa_x_list,
                                                       dfa_next_vars=self.dfa_y_list,
                                                       ts_obs_vars=self.ts_obs_list,
                                                       cudd_manager=self.manager)

                # call dijkstras for solving minimum cost path over nLTLs
                action_dict: dict = graph_search.composed_symbolic_dijkstra_nLTL(verbose=verbose)
            
            elif self.algorithm == 'astar':
                graph_search =  MultipleFormulaBDDAstar(ts_handle=self.ts_handle,
                                                        dfa_handles=self.dfa_handle_list,
                                                        ts_curr_vars=self.ts_x_list,
                                                        ts_next_vars=self.ts_y_list,
                                                        dfa_curr_vars=self.dfa_x_list,
                                                        dfa_next_vars=self.dfa_y_list,
                                                        ts_obs_vars=self.ts_obs_list,
                                                        cudd_manager=self.manager)
                # For A* we ignore heuristic computation time                                  
                start: float = time.time()
                action_dict = graph_search.composed_symbolic_Astar_search_nLTL(verbose=verbose)

            elif self.algorithm == 'bfs':
                graph_search = MultipleFormulaBFS(ts_handle=self.ts_handle,
                                                  dfa_handles=self.dfa_handle_list,
                                                  ts_curr_vars=self.ts_x_list,
                                                  ts_next_vars=self.ts_y_list,
                                                  dfa_curr_vars=self.dfa_x_list,
                                                  dfa_next_vars=self.dfa_y_list,
                                                  ts_obs_vars=self.ts_obs_list,
                                                  cudd_manager=self.manager)

                # call BFS for multiple formulas 
                action_dict: dict = graph_search.symbolic_bfs_nLTL(verbose=verbose)

            stop: float = time.time()
            print("Time took for plannig: ", stop - start)
        
        else:
            start: float = time.time()
            if self.algorithm == 'dijkstras':
                # shortest path graph search with Dijkstras
                graph_search =  SymbolicDijkstraSearch(ts_handle=self.ts_handle,
                                                       dfa_handle=self.dfa_handle_list[0],
                                                       ts_curr_vars=self.ts_x_list,
                                                       ts_next_vars=self.ts_y_list,
                                                       dfa_curr_vars=self.dfa_x_list,
                                                       dfa_next_vars=self.dfa_y_list,
                                                       ts_obs_vars=self.ts_obs_list,
                                                       cudd_manager=self.manager)

                # action_dict = graph_search.ADD_composed_symbolic_dijkstra_wLTL(verbose=False)
                action_dict = graph_search.composed_symbolic_dijkstra_wLTL(verbose=verbose)
            
            elif self.algorithm == 'astar':
                # shortest path graph search with Symbolic A*
                graph_search =  SymbolicBDDAStar(ts_handle=self.ts_handle,
                                                 dfa_handle=self.dfa_handle_list[0],
                                                 ts_curr_vars=self.ts_x_list,
                                                 ts_next_vars=self.ts_y_list,
                                                 dfa_curr_vars=self.dfa_x_list,
                                                 dfa_next_vars=self.dfa_y_list,
                                                 ts_obs_vars=self.ts_obs_list,
                                                 cudd_manager=self.manager)
                # For A* we ignore heuristic computation time                                  
                start: float = time.time()
                action_dict = graph_search.composed_symbolic_Astar_search(verbose=verbose)


            elif self.algorithm == 'bfs':
                graph_search = SymbolicSearch(ts_handle=self.ts_handle,
                                              dfa_handle=self.dfa_handle_list[0], 
                                              ts_curr_vars=self.ts_x_list,
                                              ts_next_vars=self.ts_y_list,
                                              dfa_curr_vars=self.dfa_x_list,
                                              dfa_next_vars=self.dfa_y_list,
                                              ts_obs_vars=self.ts_obs_list,
                                              cudd_manager=self.manager)

                # TODO: In future store startegy as Mealey machine (Finite State Machine)
                # The Mealey machine is a characteristic Function that represents a mapping from
                # current TS state x Obs associated with this state x State of the Automation to Next State in TS and next state in the DFA Automaton
                # TR : S_ts x Obs_bdd x S_dfa x S'_ts x S'_dfa
                action_dict = graph_search.composed_symbolic_bfs_wLTL(verbose=verbose)

            stop: float = time.time()
            print("Time took for plannig: ", stop - start)
    
        return action_dict
    

    def simulate(self, action_dict: dict,  gridworld_size: int, init_pos: tuple = (0, 0)):
        """
        A function to simulate the synthesize policy for the gridworld agent.
        """
        ts_handle = self.ts_handle
        
        ts_curr_vars = self.ts_x_list
        ts_next_vars = self.ts_y_list
        
        dfa_curr_vars = self.dfa_x_list
        dfa_next_vars = self.dfa_y_list

        if len(self.formulas) > 1:
            
            dfa_handles = self.dfa_handle_list

            if self.algorithm in ['dijkstras','astar']:
                init_state_ts_sym = ts_handle.sym_add_init_states
                state_obs_dd = ts_handle.sym_add_state_labels
                
                gridworld_strategy = convert_action_dict_to_gridworld_strategy_nLTL(ts_handle=ts_handle,
                                                                                    dfa_handles=dfa_handles,
                                                                                    action_map=action_dict,
                                                                                    init_state_ts_sym=init_state_ts_sym,
                                                                                    state_obs_dd=state_obs_dd,
                                                                                    ts_curr_vars=ts_curr_vars,
                                                                                    ts_next_vars=ts_next_vars,
                                                                                    dfa_curr_vars=dfa_curr_vars,
                                                                                    dfa_next_vars=dfa_next_vars)


                create_gridworld(size=gridworld_size, strategy=gridworld_strategy, init_pos=init_pos)
            
            else:
                # plot_policy(action_dict)

                init_state_ts_sym = ts_handle.sym_init_states
                state_obs_dd = ts_handle.sym_state_labels

                gridworld_strategy = convert_action_dict_to_gridworld_strategy_nLTL(ts_handle=ts_handle,
                                                                                    dfa_handles=dfa_handles,
                                                                                    action_map=action_dict,
                                                                                    init_state_ts_sym=init_state_ts_sym,
                                                                                    state_obs_dd=state_obs_dd,
                                                                                    ts_curr_vars=ts_curr_vars,
                                                                                    ts_next_vars=ts_next_vars,
                                                                                    dfa_curr_vars=dfa_curr_vars,
                                                                                    dfa_next_vars=dfa_next_vars)

                create_gridworld(size=gridworld_size, strategy=gridworld_strategy, init_pos=init_pos)
        else:
            dfa_handle = self.dfa_handle_list[0]

            if self.algorithm in ['dijkstras','astar']:
                init_state_ts = ts_handle.sym_add_init_states
                state_obs_dd = ts_handle.sym_add_state_labels

                gridworld_strategy = convert_action_dict_to_gridworld_strategy(ts_handle=ts_handle,
                                                                               dfa_handle=dfa_handle,
                                                                               action_map=action_dict,
                                                                               init_state_ts=init_state_ts,
                                                                               state_obs_dd=state_obs_dd,
                                                                               ts_curr_vars=ts_curr_vars,
                                                                               ts_next_vars=ts_next_vars,
                                                                               dfa_curr_vars=dfa_curr_vars,
                                                                               dfa_next_vars=dfa_next_vars)

                create_gridworld(size=gridworld_size, strategy=gridworld_strategy, init_pos=init_pos)

            else:
                init_state_ts = ts_handle.sym_init_states
                state_obs_dd = ts_handle.sym_state_labels

                gridworld_strategy = convert_action_dict_to_gridworld_strategy(ts_handle=ts_handle,
                                                                               dfa_handle=dfa_handle,
                                                                               action_map=action_dict,
                                                                               init_state_ts=init_state_ts,
                                                                               state_obs_dd=state_obs_dd,
                                                                               ts_curr_vars=ts_curr_vars,
                                                                               ts_next_vars=ts_next_vars,
                                                                               dfa_curr_vars=dfa_curr_vars,
                                                                               dfa_next_vars=dfa_next_vars)

                create_gridworld(size=gridworld_size, strategy=gridworld_strategy, init_pos=init_pos)





    def create_symbolic_lbl_vars(self,
                                 lbls,
                                 label_state_var_name: str = 'l',
                                 valid_dfa_edge_symbol_size: int = 1,
                                 add_flag: bool = False):
        """
        This function creates Boolean vairables used to create observation labels for each state. Note that in this method we do create
        prime variables as they do not switch their values.

        The number of variable you needs to represent the state observation depends on two factors.
        1. The number of objects (PDDL problem file terminology) you have (informally locations)
        2. The number of locations your observations your state observation contains
        
        e.g. Grid World: The observation associated with each state is only the location of that cell.
        So, you cannot observe l2 and l1 at the same time as our robot can only be in one cell at a given time. 

        Thus, the set of valid observation is {l1, l2}
        
        For Franka World: Say, the observation consists of the location of the robot and an object. The set of Valid locations is
        {else, l1, l2}. The set of valid observation is {(else, else), (else, l1), (else, l2), (l1, else), (l1, l1),
        (l1, l2), (l2, else) (l2, l1), (l2, l2)}

        """
        # remove skbn form the lbls as it contributes to unnecessary variables space
        try:
            del lbls['skbn']
        except KeyError:
            print("Looks like the label 'skbn' is not an object in the PDDL problem file.")


        possible_obs: list = []
        for ele in lbls:
            possible_obs.append(ele)
        # possible_obs = lbls  # for Frank world you have to update this
        if valid_dfa_edge_symbol_size > 1:
            possible_obs = list(product(possible_obs, repeat=valid_dfa_edge_symbol_size))

        state_lbl_vars: list = []
        lbl_state = label_state_var_name

        # get the number of variables in the manager. We will assign the next idex to the next lbl variables
        _num_of_sym_vars = self.manager.size()

        num_of_lbls = len(possible_obs)
        for num_var in range(math.ceil(math.log2(num_of_lbls))):
            _var_index = num_var + _num_of_sym_vars
            if add_flag:
                state_lbl_vars.append(self.manager.addVar(_var_index, f'{lbl_state}{num_var}'))
            else:
                state_lbl_vars.append(self.manager.bddVar(_var_index, f'{lbl_state}{num_var}'))

        return state_lbl_vars, possible_obs
    

    def build_bdd_abstraction(self) -> Tuple[SymbolicTransitionSystem, list, list, list]:
        """
        Main Function to Build Transition System that only represent valid edges without any weights
        """
        
        if not self.create_lbls:
            task, domain, ts_curr_state, ts_next_state  = self.create_symbolic_causal_graph(create_lbl_vars=False,
                                                                                            draw_causal_graph=False)

            sym_tr = SymbolicTransitionSystem(curr_states=ts_curr_state,
                                             next_states=ts_next_state,
                                             lbl_states=None,
                                             observations=None,
                                             task=task,
                                             domain=domain,
                                             manager=self.manager)

        else:
            task, domain, possible_obs, ts_curr_state, ts_next_state, ts_lbl_states  = self.create_symbolic_causal_graph(create_lbl_vars=True,
                                                                                                                         max_valid_formula_size=1,
                                                                                                                         draw_causal_graph=False)
            sym_tr = SymbolicTransitionSystem(curr_states=ts_curr_state,
                                              next_states=ts_next_state,
                                              lbl_states=ts_lbl_states,
                                            #   observations=possible_obs,
                                              task=task,
                                              domain=domain,
                                              manager=self.manager)

            sym_tr.create_transition_system(verbose=self.verbose, plot=self.plot_ts)
            sym_tr.create_state_obs_bdd(domain_lbls=possible_obs, verbose=self.verbose, plot=self.plot_obs)  

        return  sym_tr, ts_curr_state, ts_next_state, ts_lbl_states
    

    def create_symbolic_causal_graph(self,
                                     create_lbl_vars: bool,
                                     draw_causal_graph: bool = False,
                                     max_valid_formula_size: int = 1,
                                     add_flag: bool = False):
        """
        A function to create an instance of causal graph which call pyperplan. We access the task related properties pyperplan
        and create symbolic TR related to action.   
        """
        _causal_graph_instance = CausalGraph(problem_file=self.problem_file,
                                             domain_file=self.domain_file,
                                             draw=draw_causal_graph)

        _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)
        # print("No. of edges in the graph:", len(_causal_graph_instance.causal_graph._graph.edges()))

        task_facts = _causal_graph_instance.task.facts
        
        # the number of boolean variables (|x|) = log⌈|facts|⌉ - Because facts represent all possible predicates in our causal graph 
        curr_state, next_state = self.create_symbolic_vars(num_of_facts=len(task_facts),
                                                           add_flag=add_flag)
                                                            
        if create_lbl_vars:
            objs = _causal_graph_instance.problem.objects
            
            lbl_state, possible_obs = self.create_symbolic_lbl_vars(lbls=objs,
                                                                    valid_dfa_edge_symbol_size=max_valid_formula_size,
                                                                    add_flag=add_flag)

            return _causal_graph_instance.task, _causal_graph_instance.problem.domain, possible_obs, \
            curr_state, next_state, lbl_state

        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_state, next_state


    def build_weighted_add_abstraction(self) -> Tuple[SymbolicTransitionSystem, List[ADD], List[ADD], List[ADD]]:
        """
         Main Function to Build Transition System that represents valid edges with their corresponding weights

         Pyperplan supports the following PDDL fragment: STRIPS without action costs
        """

        # sort them according to their weights and then convert them in to addConst; reverse will sort the weights in descending order
        weight_dict = {k: v for k, v in sorted(self.weight_dict.items(), key=lambda item: item[1], reverse=True)}
        for action, w in weight_dict.items():
            weight_dict[action] = self.manager.addConst(int(w))

        task, domain, possible_obs, add_ts_curr_state, add_ts_next_state, add_ts_lbl_states  = self.create_symbolic_causal_graph(create_lbl_vars=True,
                                                                                                                                 max_valid_formula_size=1,
                                                                                                                                 draw_causal_graph=False,
                                                                                                                                 add_flag=True)
        sym_tr = SymbolicWeightedTransitionSystem(curr_states=add_ts_curr_state,
                                                  next_states=add_ts_next_state,
                                                  lbl_states=add_ts_lbl_states,
                                                  weight_dict=weight_dict,
                                                  task=task,
                                                  domain=domain,
                                                  manager=self.manager)

        sym_tr.create_weighted_transition_system(verbose=self.verbose, plot=self.plot_ts)
        sym_tr.create_state_obs_add(domain_lbls=possible_obs, verbose=self.verbose, plot=self.plot_obs)

        return  sym_tr, add_ts_curr_state, add_ts_next_state, add_ts_lbl_states
