from cudd import Cudd

from src.symbolic_graphs import SimpleGridWorld, FrankaWorld

from utls import *
from config import *


if __name__ == "__main__":
    cudd_manager = Cudd()

    if GRIDWORLD:
        # grid world files
        domain_file_path = PROJECT_ROOT + "/pddl_files/grid_world/domain.pddl"
        if OBSTACLE:
            problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}_obstacle1.pddl"
        else:
            problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"
    
        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'
        
        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }


        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=formulas,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=USE_LTLF,
                                           dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                           verbose=False,
                                           plot_ts=False,
                                           plot_obs=False,
                                           plot=False)
        
        # build the TS and DFA(s)
        gridworld_handle.build_abstraction()
        policy: dict = gridworld_handle.solve(verbose=False)
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE)

    elif FRANKAWORLD:
        #Frank World files 
        domain_file_path = PROJECT_ROOT + "/pddl_files/example_pddl/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/example_pddl/problem.pddl"

        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'

        # frankaworld stuff
        frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                         problem_file=problem_file_path,
                                         formulas=formulas,
                                         manager=cudd_manager,
                                         ltlf_flag=USE_LTLF,
                                         dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                         algorithm=algo,
                                         verbose=False,
                                         plot_ts=False,
                                         plot_obs=False,
                                         plot=False)

        # build the abstraction
        frankaworld_handle.build_abstraction()
        policy: dict = frankaworld_handle.solve(verbose=True)
    
    else:
        warnings.warn("Please set atleast one flag to True - FRANKAWORLD or GRIDWORLD!")




    
