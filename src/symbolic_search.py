'''
This file implements Symbolic Graph search algorithms
'''
import sys

from functools import reduce
from cudd import Cudd, BDD, ADD

from typing import List
from itertools import product


class SymbolicSearch(object):
    """
    Given a Graph, find the shortest path as per the symbolic A* (BDDA*) algorihtm as outlined by Jensen, Bryant, Valeso's paper.
    """

    def __init__(self, init, target, manager, curr_vars, next_vars, transition_func, trans_func_list, sym_to_state):
        self.init = init
        self.target = target
        self.manager = manager
        self.x_list = curr_vars
        self.y_list = next_vars
        # self.setup_graph()
        self.transition_fun = transition_func
        self.transition_fun_list = trans_func_list
        self.sym_to_state_map: dict = sym_to_state
        # self.transition_fun = self._build_transition_function_w_action_costs()
        # self.estimate_fun = self._build_estimate_function()
        # self.reached = []
        # self.que = []
    
    def pre_per_action(self, trans_action, From, ycube):
        """
         Compute the predecessors of 'From' under action specific transition function.

        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        fromY = From.swapVariables(self.x_list, self.y_list)
        if type(fromY) == type(self.manager.addZero()):
            _conjoin = trans_action.bddPattern() & fromY.bddPattern()
            return _conjoin.existAbstract(ycube.bddPattern()).toADD()
        else:
            return trans_action.andAbstract(fromY, ycube)
    

    def image(self, From, xcube):
        """
        Compute the set of possible state reachable from 'From' state.

        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        # check if its add or bdd
        if type(From) == type(self.manager.addZero()):
            _conjoin = self.transition_fun & From
            ImgY = _conjoin.existAbstract(xcube)
        else:
            ImgY = self.transition_fun.andAbstract(From, xcube)

        return ImgY.swapVariables(self.y_list, self.x_list)
    
    
    def image_per_action(self, trans_action, From, xcube):
        """
        Compute the set of possible state reachable from 'From' state under action specific transition function.

        andAbstract: Conjoin to another BDD and existentially quantify variables.

        If the Structures are ADD, we then convet them to BDD using bddPattern() method. We then compute the image
         (of type BDD) and then convert to add using toADD() and return the variables.
        """
        if type(From) == type(self.manager.addZero()):
            _conjoin = trans_action.bddPattern() & From.bddPattern()
            # _conjoin = trans_action & From
            ImgY = _conjoin.existAbstract(xcube.bddPattern())

            return ImgY.swapVariables(self.y_list, self.x_list).toADD()
        else:
            ImgY = trans_action.andAbstract(From, xcube)

            return ImgY.swapVariables(self.y_list, self.x_list)
    
    def _convert_cube_to_func(self, bdd_func: str) -> List[BDD]:
        """
        A helper function that convert a string of a path to its corresponding boolean form
        """

        if isinstance(bdd_func, ADD):
            bdd_func = bdd_func.bddPattern()

        bddVars = []
        for cube in bdd_func.generate_cubes():
            _amb_var = []
            var_list = []
            _idx = 0
            for count, var in enumerate(cube):
                if var == 2 and count % 2 != 0:
                    continue
                if var == 2 and count % 2 == 0:
                    _amb_var.append([self.manager.bddVar(2*_idx), ~self.manager.bddVar(2*_idx)])   # count how many vars are missing to full define the bdd
                if var == 0:
                    # use 2* idx as the variable are interleaved
                    var_list.append(~self.manager.bddVar(2*_idx))
                elif var == 1:
                    var_list.append(self.manager.bddVar(2*_idx))
                _idx += 1

            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.append(_ele[0])
                    bddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list.pop()
            else:
                bddVars.append(reduce(lambda a, b: a & b, var_list))
        return bddVars


    def symbolic_bfs(self, verbose: bool = False):
        """
        Implement a symbolic bread first search algorithm.
        """
        reached_list = []
        reached = self.init
        closed = self.manager.bddZero()
        layer = 0
        xcube = reduce(lambda x, y: x & y, self.x_list)

        reached_list.append(reached)
        while not self.target <= reached_list[layer]:

            reached_list[layer] = reached_list[layer] & ~closed

            if reached_list[layer] == self.manager.bddZero():
                print("No plan found")
                break

            closed |= reached_list[layer]
            
            # for franka world - if reached set consists only of the init set then next set of states is the
            # INTERSECTION of the image of each init state

            # if reached_list[layer].compare(self.init, 2):
            #     # first compute the individual initial states 
            #     init_states = self._convert_cube_to_func(bdd_func=reached_list[layer])
            #     # take the intersection of the states 
            #     next_state = self.manager.bddOne()
            #     for _s in init_states:
            #         print(self.sym_to_state_map.get(_s))
            #         test = self._convert_cube_to_func(self.image(From=_s, xcube=xcube))
            #         next_state = next_state & self.image(From=_s, xcube=xcube)
            #         for t in test:
            #             print(self.sym_to_state_map.get(t))

            #     reached_list.append(next_state)
            
            # else:
            reached_list.append(self.image(From=reached_list[layer], xcube=xcube))

            
            if verbose:
                # now extract the set of states that are being expanded during each iteration
                test = self._convert_cube_to_func(bdd_func=reached_list[layer])
                for _s in test:
                    _name = self.sym_to_state_map.get(_s)
                    if _name is None:
                        print('Hi!')
                        sys.exit(-1)
                    print(self.sym_to_state_map.get(_s))
                    

            layer += 1

        reached_list[layer] = reached_list[layer] & self.target

        # return self.retrieve_bfs(reached_list)
        return self.retrive_bfs_action(reached_list)

    def retrieve_bfs(self, reached_list):
        """
        Retrieve the plan from symbolic BFS algorithm
        """
        ycube = reduce(lambda a, b: a & b, self.y_list)
        plan = []
        n = len(reached_list)
        current = reached_list[n-1]
        plan.append(current)

        for layer in reversed(range(n)):
            pred = self.pre(From=current, ycube=ycube)
            if pred & reached_list[layer - 1] != self.manager.bddZero():
                current = pred & reached_list[layer - 1]
                plan.append(current)

        print(plan)
    
    def retrive_bfs_action(self, reached_list):
        """
        Retrieve the plan from symbolic BFS algorithm
        """
        ycube = reduce(lambda a, b: a & b, self.y_list)
        plan = []
        n = len(reached_list)
        current = reached_list[n - 1]

        for layer in reversed(range(n)):
            # for each action
            for idx, tran_func_action in enumerate(self.transition_fun_list):
                pred = self.pre_per_action(trans_action=tran_func_action, From=current, ycube=ycube)

                if pred & reached_list[layer - 1] != self.manager.bddZero():
                    current = pred & reached_list[layer - 1]
                    plan.append(idx)

        print(plan)

        return plan