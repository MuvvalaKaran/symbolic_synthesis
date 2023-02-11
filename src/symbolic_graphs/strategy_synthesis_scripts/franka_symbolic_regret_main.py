import re
import sys
import time
import math
import warnings


from bidict import bidict
from collections import defaultdict
from typing import Union, List, Tuple, Dict

from cudd import Cudd, BDD, ADD

from src.explicit_graphs import CausalGraph, Ltlf2MonaDFA

from src.algorithms.strategy_synthesis import AdversarialGame, GraphOfUtlCooperativeGame, GraphofBRAdvGame

from src.symbolic_graphs import ADDPartitionedDFA
from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs
from src.symbolic_graphs.hybrid_regret_graphs import HybridGraphOfUtility, HybridGraphOfBR

from src.symbolic_graphs.strategy_synthesis_scripts import FrankaRegretSynthesis
# from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld


class FrankaSymbolicRegretSynthesis(FrankaRegretSynthesis):
    """
      Main script that constructs the Main Graph in a partitioned fashion, then constructs the graph of utility (G^{u}) and
       finally the Graph of best response (G^{br}). Both these geaph will be constructed purely symbolically. 
    """

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 formulas: Union[List, str],
                 manager: Cudd,
                 algorithm: str,
                 sup_locs: List[str],
                 top_locs: List[str],
                 weight_dict: dict = {},
                 ltlf_flag: bool = True,
                 dyn_var_ord: bool = False,
                 verbose: bool = False,
                 plot_ts: bool = False,
                 plot_obs: bool = False,
                 plot_dfa: bool = False,
                 plot: bool = False,
                 create_lbls: bool = True,
                 weighting_factor: int = 1,
                 reg_factor: float = 1):
        super().__init__(domain_file=domain_file,
                         problem_file=problem_file,
                         formulas=formulas,
                         manager=manager,
                         algorithm=algorithm,
                         sup_locs=sup_locs,
                         top_locs=top_locs,
                         weight_dict=weight_dict,
                         ltlf_flag=ltlf_flag,
                         dyn_var_ord=dyn_var_ord,
                         verbose=verbose,
                         plot_ts=plot_ts,
                         plot_obs=plot_obs,
                         plot_dfa=plot_dfa,
                         plot=plot,
                         create_lbls=create_lbls)