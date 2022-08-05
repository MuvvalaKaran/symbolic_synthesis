"""
This script implements a graph search algorithm as discussed by Fabio
"""

from functools import reduce

import graphviz as gv
from cudd import Cudd
import re
import sys
import time
import math


def pre(TR, From,  ycube, x, y):
    """Compute the predecessors of From."""
    fromY = From.swapVariables(x,y)
    return TR.andAbstract(fromY, ycube)  # Conjoin to another BDD and existentially quantify variables.


def image(TR, From, xcube, x, y):
    ImgY = TR.andAbstract(From, xcube)
    return ImgY.swapVariables(y,x)


def forward_reachability(init, transition_func, x_list: list, y_list: list, plot: bool = False):
    xcube = reduce(lambda x, y: x & y, x_list)
    Reached = New = init
    while New:
        print(Reached)
        Reached.printCover()
        Img = image(TR=transition_func, From=New, xcube=xcube, x=x_list, y=y_list)
        New = Img & ~Reached   # take the intersection of image and previously unseen nodes
        Reached |= New

    print("target = ", target)

    if plot:
        m.dumpDot([Reached], file_path='reachability.dot')
        gv.Source.from_file('reachability.dot').view()


def backward_reachability(target, transition_func, x_list: list, y_list: list):
    ycube = reduce(lambda a, b: a & b, y_list)
    Z = target
    # Return the false function.
    zeta = m.bddZero()
    # Backwards Reachability
    while Z != zeta:
        zeta = Z
        # print('Z',end='')
        # Z.summary(n)
        Z |= pre(TR=transition_func, From=Z, ycube=ycube, x=x_list, y=y_list)
        Z.printCover()
        # break when you encounter init
        if Init <= Z:
            break

    # Check inclusion of initial states.
    print('BR: ', ('holds' if Init <= Z else 'does not hold'))


def graph_search(init, target, transition_func, x_list: list, y_list: list):
    """
    A naive graph search implementation. In this algorithm we backward search and forward search in the same loop to
     determine the shortest path and extract the path from the forward pass.
    """
    FReached = FNew = init
    BReached = BNew = target

    Frings = [FNew]
    Brings = [BNew]
    print('FNew: ', FNew)
    print('BNew: ', BNew)
    xcube = reduce(lambda x, y: x & y, x_list)  # creates BDD x[0] & x[1] & x[2]
    ycube = reduce(lambda a, b: a & b, y_list)  # creates BDD y[0] & y[1] & y[2]

    while FNew:
        Img = image(TR=transition_func, From=FNew, xcube=xcube, x=x_list, y=y_list)
        # Img.printCover()

        preImg = pre(TR=transition_func, From=BReached, ycube=ycube, x=x_list, y=y_list)
        BReached = BReached | preImg
        FNew = Img & ~FReached
        FReached |= FNew

        Frings.append(FNew)
        Brings.append(BReached)

        if init <= BReached:
            break

    # take intersection and store them

    # commonEle.append(FNew & BReached)
    # if target <= commonEle[-1]:
    #     print(commonEle)
    #     break
    assert len(Frings) == len(Brings), "Make sure length of the froward pass and backward pass are the same"

    commonEle = []
    for i in range(len(Frings)):
        idx = i
        back_idx = len(Frings) - i - 1
        commonEle.append(Frings[idx] & Brings[back_idx])
        commonEle[-1].printCover()


def bellman_ford_algo(init, target, transition_func, x_list: list, y_list: list):
    """
    Symbolic implementation of bellman-ford algorithm as discussed in Fabio's paper.
    """
    pass


class BDDA_star:
    """
    Given a Graph, find the shortest path as per the symbolic A* (BDDA*) algorihtm as outlined by
     Jensen, Bryant, Valeso's paper.
    """

    def __init__(self):
        self.init = None
        self.target = None
        self.manager = Cudd()
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.setup_graph()
        # self.transition_fun = self._build_transition_function_w_actions()
        self.transition_fun = self._build_transition_function_w_action_costs()
        self.estimate_fun = self._build_estimate_function()
        self.reached = []
        self.que = []

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

    def pre(self, From, ycube):
        """
        Compute the predecessors of 'From'.
        
        andAbstract: Conjoin to another BDD and existentially quantify variables.
        """
        fromY = From.swapVariables(self.x_list, self.y_list)
        return self.transition_fun.andAbstract(fromY, ycube)

    def strong_pre(self, From, ycube):
        """
        Compute the predecessors of 'From' from which you can force a visit to the 'From'.

        univAbstract: Universally quantify variables from this BDD
        """
        fromY = From.swapVariables(self.x_list, self.y_list)
        return ~self.transition_fun.andAbstract(~fromY, ycube)

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

    def _build_transition_function_w_action_costs(self):
        """
        Build the transition function corresponding to the gunslinger-mib (Man in black) problem from Kissmann's thesis
        """
        # create variables and corresponding constants. have 10 states, so 4 boolean vairables
        # with 16 - 10 = 6 unused ones
        num_vars = 4
        self.x_list = [self.manager.addVar(2*i, 'x' + str(i)) for i in range(num_vars)]
        self.y_list = [self.manager.addVar(2*i + 1, 'y' + str(i)) for i in range(num_vars)]
        x = self.x_list
        y = self.y_list

        # then build the chunks of transition system with same action cost.
        # For us, each action types has a constant value associated with it.
        # P0 = ~x[0] & ~x[1] & ~x[2] & ~x[3]
        # P1 = x[0] & ~x[1] & ~x[2] & ~x[3]
        # P2 = ~x[0] & x[1] & ~x[2] & ~x[3]
        # P3 = ~x[0] & ~x[1] & x[2] & ~x[3]
        # P4 = ~x[0] & ~x[1] & ~x[2] & x[3]
        # P5 = x[0] & x[1] & ~x[2] & ~x[3]
        # P6 = x[0] & ~x[1] & x[2] & ~x[3]
        # P7 = x[0] & ~x[1] & ~x[2] & x[3]
        # P8 = ~x[0] & x[1] & x[2] & ~x[3]
        # P9 = ~x[0] & x[1] & ~x[2] & x[3]

        # a = 1; b = 2; c = 3; d = 4; e = 5; f = 6; g = 7; h = 10; i = 12; j = 13
        c_a = self.manager.addConst(int(1))
        c_b = self.manager.addConst(int(2))
        c_c = self.manager.addConst(int(3))
        c_d = self.manager.addConst(int(4))
        c_e = self.manager.addConst(int(5))
        c_f = self.manager.addConst(int(6))
        c_g = self.manager.addConst(int(7))
        c_h = self.manager.addConst(int(10))
        c_i = self.manager.addConst(int(12))
        c_j = self.manager.addConst(int(13))

        # c_a = self.manager.addConst(int(1))
        # c_b = self.manager.addConst(int(1))
        # c_c = self.manager.addConst(int(1))
        # c_d = self.manager.addConst(int(1))
        # c_e = self.manager.addConst(int(1))
        # c_f = self.manager.addConst(int(1))
        # c_g = self.manager.addConst(int(1))
        # c_h = self.manager.addConst(int(1))
        # c_i = self.manager.addConst(int(1))
        # c_j = self.manager.addConst(int(10))

        TR_a = (c_a & ((~x[0] & ~x[1] & ~x[2] & ~x[3] & ~y[0] & y[1] & ~y[2] & ~y[3]) |   # p0 -> p2
                (x[0] & ~x[1] & x[2] & ~x[3] & y[0] & ~y[1] & ~y[2] & y[3]) |   # p6 -> p7
                (x[0] & ~x[1] & x[2] & ~x[3] & ~y[0] & y[1] & y[2] & ~y[3])))    # p6 -> p8

        TR_b = (c_b & ((~x[0] & ~x[1] & ~x[2] & ~x[3] & y[0] & ~y[1] & ~y[2] & ~y[3]) |  # p0 -> p1
                (x[0] & ~x[1] & ~x[2] & ~x[3] & ~y[0] & ~y[1] & y[2] & ~y[3]) |  # p1 -> p3
                (~x[0] & ~x[1] & x[2] & ~x[3] & y[0] & y[1] & ~y[2] & ~y[3])))   # p3 -> p5

        TR_c = (c_c & ((~x[0] & ~x[1] & x[2] & ~x[3] & ~y[0] & ~y[1] & ~y[2] & y[3]) |  # p3 -> p4
                (x[0] & x[1] & ~x[2] & ~x[3] & ~y[0] & y[1] & y[2] & ~y[3]) |  # p5 -> p8
                (x[0] & ~x[1] & ~x[2] & x[3] & ~y[0] & y[1] & ~y[2] & y[3])))  # p7 -> p9

        TR_d = (c_d & ~x[0] & x[1] & ~x[2] & ~x[3] & ~y[0] & ~y[1] & y[2] & ~y[3])  # p2 -> p3

        TR_e = (c_e & ~x[0] & ~x[1] & ~x[2] & x[3] & y[0] & ~y[1] & ~y[2] & y[3])  # p4 -> p7

        TR_f = (c_f & ~x[0] & ~x[1] & ~x[2] & x[3] & y[0] & ~y[1] & y[2] & ~y[3])  # p4 -> p6

        TR_g = (c_g & ((x[0] & ~x[1] & ~x[2] & ~x[3] & ~y[0] & ~y[1] & ~y[2] & y[3]) |  # p1 -> p4
                (~x[0] & ~x[1] & x[2] & ~x[3] & y[0] & ~y[1] & y[2] & ~y[3]) |  # p3 -> p6
                (x[0] & x[1] & ~x[2] & ~x[3] & y[0] & ~y[1] & y[2] & ~y[3])))   # p5 -> p6

        TR_h = (c_h & ((~x[0] & ~x[1] & ~x[2] & ~x[3] & ~y[0] & ~y[1] & y[2] & ~y[3]) |  # p0 -> p3
                (~x[0] & x[1] & x[2] & ~x[3] & ~y[0] & y[1] & ~y[2] & y[3])))  # p8 -> p9

        TR_i = (c_i &~x[0] & x[1] & ~x[2] & ~x[3] & y[0] & y[1] & ~y[2] & ~y[3])  # p2 -> p5

        TR_j = (c_j & x[0] & ~x[1] & x[2] & ~x[3] & ~y[0] & y[1] & ~y[2] & y[3])  # p6 -> p9

        self.init = ~x[0] & ~x[1] & ~x[2] & ~x[3]
        self.target = ~x[0] & x[1] & ~x[2] & x[3]

        self.transition_fun_list = [TR_a, TR_b, TR_c, TR_d, TR_e, TR_f, TR_g, TR_h, TR_i, TR_j]
        self.transition_costs = [c_a, c_b, c_c, c_d, c_e, c_f, c_g, c_h, c_i, c_j]

        return TR_a | TR_b | TR_c | TR_d | TR_e | TR_f | TR_g | TR_h | TR_i | TR_j

    def _build_transition_function_w_actions(self):
        """
        Build the transition function with actions embedded in it.
        """
        # manually create inter-leaved variables
        x0 = self.manager.bddVar(0, 'x0')
        y0 = self.manager.bddVar(1, 'y0')
        x1 = self.manager.bddVar(2, 'x1')
        y1 = self.manager.bddVar(3, 'y1')

        self.x_list = [x0, x1]
        self.y_list = [y0, y1]

        TR_a =((~self.x_list[0] & ~self.x_list[1] & self.y_list[0] & ~self.y_list[1]) |
             (~self.x_list[0] & ~self.x_list[1] & ~self.y_list[0] & self.y_list[1]))

        TR_c = (~self.x_list[0] & self.x_list[1] & self.y_list[0] & self.y_list[1])

        TR_b = (self.x_list[0] & ~self.x_list[1] & ~self.y_list[0] & self.y_list[1])

        TR_d = (self.x_list[0] & ~self.x_list[1] & self.y_list[0] & self.y_list[1])

        # TR_d = ((self.x_list[0] & self.x_list[1] & self.y_list[0] & ~self.y_list[1]))

        self.init = ~x0 & ~x1
        self.target = x0 & x1

        self.transition_fun_list = [TR_a, TR_b, TR_c, TR_d]

        return TR_a | TR_b | TR_c | TR_d

    def _build_transition_function(self):
        # manually create inter-leaved variables
        x0 = self.manager.bddVar(0, 'x0')
        y0 = self.manager.bddVar(1, 'y0')
        x1 = self.manager.bddVar(2, 'x1')
        y1 = self.manager.bddVar(3, 'y1')

        self.x_list = [x0, x1]
        self.y_list = [y0, y1]

        TR = ((~x0 & ~x1 & y0 & ~y1) |
              (~x0 & ~x1 & ~y0 & y1) |
              (~x0 & x1 & y0 & y1) |
              (x0 & x1 & y0 & ~y1))

        # testing by adding a diagonal
        TR = ((~x0 & ~x1 & y0 & ~y1) |
              (~x0 & ~x1 & ~y0 & y1) |
              (~x0 & x1 & y0 & y1) | (~x0 & x1 & ~y0 & ~y1) |
              (x0 & ~x1 & ~y0 & y1) |    # added diagonal transition
              (x0 & ~x1 & y0 & y1))

        self.init = ~x0 & ~x1
        self.target = x0 & x1

        return TR

    def _build_estimate_function(self):
        # lets have a different bdd for each h value; i
        z0 = self.z_list[0]
        z1 = self.z_list[1]
        h1 = ((~z0 & ~z1) |
              (z0 & ~z1))

        h0 = ((~z0 & z1) |
              (z0 & z1))

        return [h0, h1]

    def setup_graph(self):

        # we need to create the estimate function
        c0 = self.manager.addConst(float(0))
        c1 = self.manager.addConst(float(1))
        # z = [mgr.addVar(4+i, 'x' + str(i)) for i in range(2)]
        # z0 = mgr.addVar(4, 'z0')
        # z1 = mgr.addVar(5, 'z1')

        # for now try with bdd
        z0 = self.manager.bddVar(4, 'z0')
        z1 = self.manager.bddVar(5, 'z1')
        self.z_list = [z0, z1]


    def _get_h_val(self, state) -> list:
        val_list = []
        for h_val, heur in enumerate(self.estimate_fun):
            # first swap variables
            heur = heur.swapVariables(self.z_list, self.x_list)
            intersection_states = heur & state
            if intersection_states:
                val_list.append((h_val, intersection_states))

        return val_list

    def _get_improvement(self, curr_state, next_state) -> list:
        """
        A function that returns the improvement value under a valid transition
        """
        impr_list = []

        curr_state_val_list = self._get_h_val(curr_state)
        next_state_val_list = self._get_h_val(next_state)

        # for every curr state, check if it can transit to the next state,
        # if yes, compute improvement and store the value along with the next state value
        for c_state_val, c_state in curr_state_val_list:
            for n_state_val, n_state, in next_state_val_list:
                # check if its valid transition
                if self.transition_fun & c_state & n_state.swapVariables(self.x_list, self.y_list):
                    impr_list.append([c_state_val - n_state_val, n_state])

        return impr_list

        # curr_state_val = math.inf
        # next_state_val = math.inf
        # for h_val, heur in enumerate(estimates):
        #     # first swap variables
        #     heur = heur.swapVariables(z_list, x_list)
        #     intersection_curr_states = heur & curr_state
        #     intersection_next_states = heur & next_state
        #
        #     if intersection_curr_states:
        #         curr_state_val = h_val
        #     if intersection_next_states:
        #         next_state_val = h_val
        #
        #     if curr_state_val != math.inf and next_state_val != math.inf:
        #         break

        # return 0

    def _sort_nodes(self, unsoretd_nodes):
        """
        Sort the nodes added to the Que based on f value. If two BDDs have same f value, then sort by their h value
        """
        sorted_nodes_f = sorted(unsoretd_nodes, key=lambda unsoretd_nodes: (unsoretd_nodes[0], unsoretd_nodes[2] ), reverse=True)

        # now sort as per the h value to resolve ties amongst f values
        # sorted_nodes_f_h = sorted(sorted_nodes_f, key=lambda sorted_nodes_f: sorted_nodes_f[2])
        # print(sorted_nodes_f_h)

        return sorted_nodes_f

    def _get_top_node(self, top_node):
        """
        A helper function to retrieve the BDD corresponding to the top node
        """
        if type(top_node) == tuple:
            return top_node
        else:
            return top_node[0]

    def _check_goal_on_top(self, top_node, goal) -> bool:
        """
        A helper function to check if goal state is at the top of the queue or not

        @Return: true if goal <= node else false
        """
        if type(top_node) == tuple:
            return goal <= top_node[-1]
        else:
            return goal <= top_node[0][-1]

    def _compute_unseen_nodes(self, image, reached_set):
        """
        A helper function to prune seen nodes and compute the image of previously unseen nodes
        """
        if type(reached_set) == tuple:
            return image & ~reached_set[-1]
        else:
            return image & ~reached_set[-1][-1]

    def get_path(self):
        """
        Perform Backwas reachbility and at intersection at each iteration with reached
        """
        ycube = reduce(lambda a, b: a & b, self.y_list)
        Z = self.target
        # Return the false function
        zeta = self.manager.bddZero()
        path = [Z]
        # Backwards Reachability
        itr = 1
        while Z != zeta:
            zeta = Z
            Z |= self.pre(From=Z, ycube=ycube)
            itr += 1
            path.append(Z & self.reached[-1* itr][-1][-1])
            # break when you encounter init
            if self.init <= Z:
                break

        print(path)

    def solve(self, weight):
        """
        Implement SetA* algorithm as discussed by Jensen, Bryant, Valeso's paper.

        TODO:
         1. Fix the append method where you append states with same g,h value to the same node in the Que
         2. Check the improvement function
         3. Add TR partitioning for speed ups
        """
        #
        # def get_h_val(estimates, state) -> list:
        #     val_list = []
        #     for h_val, heur in enumerate(estimates):
        #         # first swap variables
        #         heur = heur.swapVariables(z_list, x_list)
        #         intersection_states = heur & state
        #         if intersection_states:
        #             val_list.append((h_val, intersection_states))
        #
        #     return val_list
        #
        # def get_improvement(estimates, curr_state, next_state, transition, x_list, y_list) -> list:
        #     """
        #     A function that returns the improvement value under a valid transition
        #     """
        #     impr_list = []
        #
        #     curr_state_val_list = get_h_val(estimates, curr_state)
        #     next_state_val_list = get_h_val(estimates, next_state)
        #
        #     # for every curr state, check if it can transit to the next state,
        #     # if yes, compute improvement and store the value along with the next state value
        #     for c_state_val, c_state in curr_state_val_list:
        #         for n_state_val, n_state, in next_state_val_list:
        #             # check if its valid transition
        #             if transition & c_state & n_state.swapVariables(x_list, y_list):
        #                 impr_list.append([c_state_val - n_state_val, n_state])
        #
        #     return impr_list
        #
        #     # curr_state_val = math.inf
        #     # next_state_val = math.inf
        #     # for h_val, heur in enumerate(estimates):
        #     #     # first swap variables
        #     #     heur = heur.swapVariables(z_list, x_list)
        #     #     intersection_curr_states = heur & curr_state
        #     #     intersection_next_states = heur & next_state
        #     #
        #     #     if intersection_curr_states:
        #     #         curr_state_val = h_val
        #     #     if intersection_next_states:
        #     #         next_state_val = h_val
        #     #
        #     #     if curr_state_val != math.inf and next_state_val != math.inf:
        #     #         break
        #
        #     # return 0
        #
        # def sort_nodes(unsoretd_nodes):
        #     """
        #     Sort the nodes added to the Que based on f value. If two BDDs have same f value, then sort by their h value
        #     """
        #     sorted_nodes_f = sorted(unsoretd_nodes, key=lambda unsoretd_nodes: unsoretd_nodes[0])
        #
        #     # now sort as per the h value to resolve ties amongst f values
        #     # sorted_nodes_f_h = sorted(sorted_nodes_f, key=lambda sorted_nodes_f: sorted_nodes_f[2])
        #     # print(sorted_nodes_f_h)
        #
        #     return sorted_nodes_f
        #
        # def get_top_node(top_node):
        #     """
        #     A helper function to retrieve the BDD corresponding to the top node
        #     """
        #     if type(top_node) == tuple:
        #         return top_node
        #     else:
        #         return top_node[0]
        #
        # def check_goal_on_top(top_node, goal) -> bool:
        #     """
        #     A helper function to check if goal state is at the top of the queue or not
        #
        #     @Return: true if goal <= node else false
        #     """
        #     if type(top_node) == tuple:
        #         return goal <= top_node[-1]
        #     else:
        #         return goal <= top_node[0][-1]
        #
        # def compute_unseen_nodes(image, reached_set):
        #     """
        #     A helper function to prune seen nodes and compute the image of previously unseen nodes
        #     """
        #     if type(reached_set) == tuple:
        #         return image & ~reached_set[-1]
        #     else:
        #         return image & ~reached_set[-1][-1]

        g_val: int = 0

        h_vals = self._get_h_val(self.init)
        if len(h_vals) == 1:
            h_val = h_vals[0][0]
            h_state = h_vals[0][1]

            assert h_state == self.init, "Your estimate function is fucked up. Fix it"
        else:
            print("Starting from multiple initial ")
        f_val = (1 - weight)*g_val + weight*h_val
        self.que = [(f_val, g_val, h_val, self.init)]
        self.reached = [[(g_val, h_state)]]
        xcube = reduce(lambda x, y: x & y, self.x_list)

        while self.que and not self._check_goal_on_top(self.que[-1], self.target):
            curr_node = self._get_top_node(self.que.pop())
            next = self.image(From=curr_node[-1], xcube=xcube)
            new = self._compute_unseen_nodes(image=next, reached_set=self.reached[-1])

            h_vals = self._get_improvement(curr_state=curr_node[-1],
                                           next_state=new)
            g_val += 1
            node_tuples = []
            curr_h_val = curr_node[2]
            for h_value, h_state in h_vals:

                h_val = curr_h_val - h_value
                f_val = (1 - weight)*g_val + weight*h_val
                node_tuples.append((f_val, g_val, h_val, h_state))

            # append the nodes and then sort

            for n in node_tuples:
                self.que.insert(0, n)
            self.que = self._sort_nodes(unsoretd_nodes=self.que)
            self.reached.append([(g_val, new)])

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

    def symbolic_bfs(self):
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

            closed |= reached_list[layer]
            reached_list.append(self.image(From=reached_list[layer], xcube=xcube))
            layer += 1

        reached_list[layer] = reached_list[layer] & self.target

        # return self.retrieve_bfs(reached_list)
        return self.retrive_bfs_action(reached_list)

    def retrieve_bidirectional_bfs_action(self, freached_list, breached_list):
        """
        Retrieve the plan from symbolic BFS algorithm
        """
        ycube = reduce(lambda a, b: a & b, self.y_list)
        xcube = reduce(lambda x, y: x & y, self.x_list)
        fplan = []
        f_layer = len(freached_list)
        current = freached_list[f_layer - 1]

        for layer in reversed(range(f_layer)):
            # for each action
            for idx, tran_func_action in enumerate(self.transition_fun_list):
                pred = self.pre_per_action(trans_action=tran_func_action, From=current, ycube=ycube)

                if pred & freached_list[layer - 1] != self.manager.bddZero():
                    current = pred & freached_list[layer - 1]
                    fplan.append(idx)
        print(fplan)

        # start finding the intermediate state at the intersection of forward and backward reachability
        inter = self.init
        for step in range(f_layer - 1):
            trans_fnc_action = self.transition_fun_list[fplan[step]]
            inter = self.image_per_action(trans_action=trans_fnc_action, From=inter, xcube=xcube)

        print("The intermediate node of the two frontiers is: ", inter)

        bplan = []
        b_layer = len(breached_list)
        current = inter

        for layer in range(b_layer - 1):
            # for each action
            for idx, tran_func_action in enumerate(self.transition_fun_list):
                succ = self.image_per_action(trans_action=tran_func_action, From=current, xcube=xcube)

                if succ & breached_list[layer] != self.manager.bddZero():
                    current = succ & breached_list[layer]
                    bplan.append(idx)
        print(bplan)

    def symbolic_bidirectional_bfs(self):
        """
        Implement a symbolic bidirectional plan
        """
        freached_list = []
        breached_list = []
        freach = self.init
        breach = self.target
        fclosed = bclosed = self.manager.bddZero()
        flayer = blayer = 0
        ftime = btime = 0

        freached_list.append(freach)
        breached_list.append(breach)

        xcube = reduce(lambda x, y: x & y, self.x_list)
        ycube = reduce(lambda a, b: a & b, self.y_list)

        while freached_list[flayer] & breached_list[blayer] == self.manager.bddZero():
            if ftime <= btime:
                fstart = time.time()
                freached_list[flayer] = freached_list[flayer] & ~fclosed

                if freached_list[flayer] == self.manager.bddZero():
                    print("No plan found")

                fclosed |= freached_list[flayer]
                freached_list.append(self.image(From=freached_list[flayer], xcube=xcube))
                flayer += 1
                fstop = time.time()
                ftime = fstop - fstart
            else:
                bstart = time.time()
                breached_list[blayer] = breached_list[blayer] & ~bclosed

                if breached_list[blayer] == self.manager.bddZero():
                    print("No plan found")

                bclosed |= breached_list[blayer]
                breached_list.append(self.pre(From=breached_list[blayer], ycube=ycube))
                blayer += 1
                bstop = time.time()
                btime = bstop - bstart

        freached_list[flayer] = freached_list[flayer] & breached_list[flayer]
        breached_list[blayer] = freached_list[flayer] & breached_list[flayer]

        print(freached_list)
        print(breached_list)

        return self.retrieve_bidirectional_bfs_action(freached_list, breached_list)

    def retrieve_dijkstra(self, max_layer, bdd_freach_list):
        """
        Retrieval of a plan for symbolic Dijkstra
        """
        plan = []
        current = bdd_freach_list[max_layer]
        ycube = reduce(lambda a, b: a & b, self.y_list)
        g_layer = self.manager.addConst(int(max_layer))

        while (self.init & current).isZero():
            for tr_idx, tr_action in reversed(list(enumerate(self.transition_fun_list))):
                pred = self.pre_per_action(trans_action=tr_action,
                                           From=current,
                                           ycube=ycube)
                # first get the corresponding transition action cost (constant at the terminal node)
                action_cost_cnst = tr_action.findMax()
                step = g_layer - action_cost_cnst
                if step.isZero():
                    step_val = 0
                else:
                    step_val = int(re.findall(r'\d+', step.__repr__())[0])
                if not (bdd_freach_list[step_val] & pred).isZero():
                    current = bdd_freach_list[step_val] & pred
                    plan.append(tr_idx)

                    g_layer = step

        print(plan)

    def symbolic_dijkstra(self):
        """
        Implement a general action cost planning algorithm using Dijkstra's algorithm in a symbolic fashion.

        To represent the cost along with an action, we use ADD instead of BDDs.

        NOTE: If we were to represent edge cost as boolean variables with max cost of C_max, then we would have
         required Upper bounds(log2(C_max + 1)) boolean variables. So, for a tiny graph with very large wieghts this
         would scale poorly. Instead we use ADDs to mitigate this problem as we only need as many constant variables
         as the numbers of actions costs in the causal graph (graph over which we are planning).
        """

        # TODO: In future change this
        open_list = [self.manager.addZero() for _ in range(19 + 13)]

        xcube = reduce(lambda x, y: x & y, self.x_list)

        closed = self.manager.addZero()
        g_val = self.manager.addZero()

        if g_val.isZero():
            g_layer = 0
        else:
            print("Error: Dijkstra's algorithm should always start from the 0th layer. Fix Code!!!!!")
            sys.exit(-1)

        open_list[g_layer] = self.init
        while not self.target <= open_list[g_layer]:
            open_list[g_layer] = open_list[g_layer] & ~closed

            # we will implement the if loop later

            # expand the unexplored states
            if open_list[g_layer] != self.manager.addZero():
                closed |= open_list[g_layer]

                # use .bddPattern() to extract the bdd out of the ADD
                # for tr_idx, action_cost_cnst in enumerate(self.transition_costs):
                #     step = g_val + action_cost_cnst
                #     step_val = int(re.findall(r'\d+', step.__repr__())[0])
                #     open_list[step_val] |= self.image_per_action(trans_action=self.transition_fun_list[tr_idx],
                #                                                  From=open_list[g_layer],
                #                                                  xcube=xcube)

                for tr_action in self.transition_fun_list:
                    # first get the corresponding transition action cost (constant at the terminal node)
                    action_cost = tr_action.findMax()
                    step = g_val + action_cost
                    step_val = int(re.findall(r'\d+', step.__repr__())[0])
                    image_c = self.image_per_action(trans_action=tr_action,
                                                    From=open_list[g_layer],
                                                    xcube=xcube)
                    open_list[step_val] = open_list[step_val] | image_c

            g_val = g_val + self.manager.addOne()
            g_layer += 1

        open_list[g_layer] = open_list[g_layer] & self.target

        return self.retrieve_dijkstra(max_layer=g_layer, bdd_freach_list=open_list)





def testing():
    """
     A graveyard of useless trail and error coce
    """
    mgr = Cudd()

    l = 4  # create a, b, c, d as four actions
    # a, b, c, d = (mgr.addVar(None, nm) for nm in ['a', 'b', 'c', 'd'])

    a = mgr.addConst(int(1))
    b = mgr.addConst(int(2))
    c = mgr.addConst(int(3))
    d = mgr.addConst(int(4))

    # manually create inter-leaved variables
    x0 = mgr.addVar(4, 'x0')
    y0 = mgr.addVar(5, 'y0')
    x1 = mgr.addVar(6, 'x1')
    y1 = mgr.addVar(7, 'y1')

    T = ((a & ~x0 & ~x1 & y0 & ~y1) |
         (b & ~x0 & ~x1 & ~y0 & y1) |
         (c & ~x0 & x1 & y0 & y1) |
         (d & x0 & x1 & y0 & ~y1))

    T = (~T).ite(m.plusInfinity(), T)

    T.display()

    init = ~x0 & ~x1

    conjoin = T & init
    ImgY = conjoin.existAbstract(x0 & x1)
    # ImgY = T.andAbstract(From, xcube)
    ImgX = ImgY.swapVariables([y0, y1], [x0, x1])
    ImgX.display()

    # g = b
    # T_check = T.univAbstract(g)
    # T_check.printCover()
    #
    # T_check = T.existAbstract(g)
    # T_check.printCover()
    #
    # r = T.restrict(g)
    # r.printCover()
    # print(r)

    # mgr.dumpDot([r], file_path='reachability.dot')
    # gv.Source.from_file('reachability.dot').view()
    # T_check = T.univAbstract(~y[0])
    # print(T_check)
    # T_check = T.existAbstract(x[0])
    # print(T_check)


if __name__ == "__main__":
    m = Cudd()
    n = 2
    x = [m.bddVar(i, 'x' + str(i)) for i in range(n)]
    y = [m.bddVar(n + i, 'y' + str(i)) for i in range(n)]
    T = ((~x[0] & ~x[1] & y[0] & ~y[1]) |
         (~x[0] & ~x[1] & ~y[0] & y[1]) |
         (~x[0] & x[1] & y[0] & y[1]) |
         (x[0] & x[1] & y[0] & ~y[1]))

    # Current- and next-state variables.
    n = 3
    x = [m.bddVar(i, 'x' + str(i)) for i in range(n)]    # use bddVar to create bdd Variables
    y = [m.bddVar(n + i, 'y' + str(i)) for i in range(n)]

    # Auxiliary cube for preimage computation.
    # ycube = y[2] & y[1] & y[0]

    # Create the transition function
    TR = ((~x[0] & ~x[1] & ~x[2] & y[0] & y[1] & ~y[2]) |
          (~x[0] & ~x[1] & ~x[2] & y[0] & ~y[1] & ~y[2]) |
          (~x[0] & ~x[1] & ~x[2] & y[0] & ~y[1] & y[2]) |
          (~x[0] & ~x[1] & ~x[2] & ~y[0] & ~y[1] & y[2]) |
          (x[0] & ~x[1] & ~x[2] & ~y[0] & y[1] & ~y[2]) |
          (~x[0] & x[1] & ~x[2] & ~y[0] & ~y[1] & ~y[2]) |
          (~x[0] & ~x[1] & x[2] & ~y[0] & ~y[1] & y[2]) | (~x[0] & ~x[1] & x[2] & ~y[0] & ~y[1] & ~y[2]) |
          (x[0] & x[1] & ~x[2] & ~y[0] & y[1] & y[2]) |
          (x[0] & ~x[1] & x[2] & ~y[0] & y[1] & y[2]) |
          (~x[0] & x[1] & x[2] & y[0] & ~y[1] & ~y[2]) | (~x[0] & x[1] & x[2] & y[0] & y[1] & y[2]) |
          (x[0] & x[1] & x[2] & ~y[0] & y[1] & y[2])
          )
    Init = ~x[0] & ~x[1] & ~x[2]  # 1
    target = ~x[0] & x[1] & x[2]  # 7

    print("TR = ", TR)
    print("Init = ", Init)

    # forward reachability
    forward_reachability(init=Init, transition_func=TR, x_list=x, y_list=y)

    # backwards reachability
    backward_reachability(target=target, transition_func=TR, x_list=x, y_list=y)

    # graph search
    graph_search(init=Init, target=target, transition_func=TR,  x_list=x, y_list=y)

    # testing()
    # sys.exit(-1)
    # create an instance of BDDA* class and find the shortest path
    # bdd_set_a_star = BDDA_star()
    # bdd_set_a_star.solve(weight=0.5)
    # bdd_set_a_star.get_path()

    # do BFS
    bdd_bfs = BDDA_star()
    # bdd_bfs.symbolic_bfs()
    #
    # bdd_bfs.symbolic_bidirectional_bfs()

    bdd_bfs.symbolic_dijkstra()
    
    # get preimage of s3 
    # x_list = bdd_set_a_star.x_list
    # y_list = bdd_set_a_star.y_list
    # ycube = reduce(lambda a, b: a & b, y_list)
    # pre = bdd_set_a_star.strong_pre(From=(x_list[0] & x_list[1]), ycube=ycube)
    # print(pre)