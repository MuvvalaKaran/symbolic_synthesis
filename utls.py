'''
Miscellaneous utility functions go in here
'''
import warnings

from typing import List

from src.explicit_graphs import CausalGraph
from src.explicit_graphs import TwoPlayerGame
from src.explicit_graphs import FiniteTransitionSystem

def get_graph(domain_file: str,
              problem_file: str,
              formulas: List[str] = '',
              print_flag: bool = False,
              plot_ts: bool =False,
              plot_dfa: bool = False,
              plot_product: bool = False) -> None:
    """
    A helepr function to plaot the causal graphs and the DFAs. The product graph is a work under construction

    Note on Causal Graph: For Gridworld, the # of states is directly proportionaly to the # of grid cells.
     So, a m by m gridworld will have exaclt m^2 states. Now if oyu have DFA with n states, then the product graph will have
     m*m*n states. For k DFAs each with n states, we will have in total = k*m*m*n states in the product graph.
    
    Unlike the manipulation domain, the # of states in the product graph does not explode exponentially.
    
    For e.g. for 20 by 20 gridworld, 5 DFAs each with 5 states, we have 10,000 States in our Product graph.
    
    """

    _causal_graph_instance = CausalGraph(problem_file=problem_file,
                                         domain_file=domain_file,
                                         draw=plot_ts)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.edges())}")
    
    _two_player_instance = TwoPlayerGame(_causal_graph_instance._causal_graph, None)

    # # Build the automaton
    # if formulas == '':
    #     _dfa = _two_player_instance.build_LTL_automaton(formula="F((l8 & l9 & l0) || (l3 & l2 & l1))", plot=plot_dfa)
    #     _product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                                         trans_sys=_causal_graph_instance._causal_graph,
    #                                                         absorbing=True,
    #                                                         plot=plot_product)
    
    # # now lets plot a product of two DFAs
    # if len(formulas) > 1:
    #     formula_list_len = len(formulas) - 1
    #     for idx, ltl in enumerate(formulas):
    #         _dfa = _two_player_instance.build_LTL_automaton(formula=ltl, plot=plot_dfa)
    #         if idx == 0:
    #             _product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                             trans_sys=_causal_graph_instance._causal_graph,
    #                                             absorbing=True,
    #                                             plot=False)
    #         elif idx == formula_list_len:
    #             _product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                             trans_sys=_product_graph,
    #                                             absorbing=True,
    #                                             plot=plot_product)
    #         else:
    #             _product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                             trans_sys=_product_graph,
    #                                             absorbing=True,
    #                                             plot=False)
    # else:
    #     _dfa = _two_player_instance.build_LTL_automaton(formula=formulas[0], plot=plot_dfa)
    #     _product_graph = _two_player_instance.build_product(dfa=_dfa,
    #                                                         trans_sys=_causal_graph_instance._causal_graph,
    #                                                         absorbing=True,
    #                                                         plot=plot_product)
    
    # if print_flag:
    #     print(f"No. of nodes in the product graph is :{len(_product_graph._graph.nodes())}")
    #     print(f"No. of edges in the product graph is :{len(_product_graph._graph.edges())}")

    # print("Done building the Product Automaton")

# A decorator to throw warning when we use deprecated methods/functions/routines
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func