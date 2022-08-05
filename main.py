import os
import sys

from dd.autoref import BDD
from src.causal_graph import CausalGraph
from src.two_player_game import TwoPlayerGame
from src.transition_system import FiniteTransitionSystem


def test_constraint_vs_explicit_transition_thr():
    """
    A method to  test my speculation on the elevator reachability problem

    Floor 0: 00 (!x0.!x1)
    Floor 2: 01 (!x0.x1)
    Floor 1: 10 (x0.!x1)
    Floor 3: 11 (x0.x1)

    We use prime to represent the next state we can transit to.

    Floor 0: 00 (!x0'.!x1')
    Floor 2: 01 (!x0'.x1')
    Floor 1: 10 (x0'.!x1')
    Floor 3: 11 (x0'.x1')

    """
    # instantiate a manger
    bdd = BDD()
    # bdd.declare('x')
    # bdd.add_expr(r'~ x')
    # bdd.dump("single_var.pdf")
    # sys.exit(-1)


    # declare variables
    bdd.declare("x0", "x0'", "x1", "x1'")
    # TLA+ syntax
    # only adding the constraints - the set of valid assignments correspond to the set of valid transitions
    s = (
        r"((~ x0 /\ ~ x1) => ( (~ x0' /\ ~ x1') \/ (x0' /\ ~ x1') )) /\ "
        r"((x0 /\ ~ x1) => ~ (x0' /\ x1')) /\ "
        r"((~ x0 /\ x1) => ( (~ x0' /\ x1') \/ (x0' /\ ~ x1') )) /\ "
        r" ~ (x0 /\ x1)")

    # explicitly stating the transitions
    s_expl = r"(!x0 & !x1 & x0' & !x1') |" \
             r" (!x0 & x1 & x0' & !x1') |" \
             r" (x0 & !x1 & !x0' & x1') |" \
             r" (x0 & !x1 & !x0' & !x1')"
    transitions = bdd.add_expr(s_expl)

    # print(transitions.negated)

    # to count the # of satifiable assignments and the assignmet that lead to valid transition
    print(bdd.count(~ transitions))
    print(list(bdd.pick_iter(~ transitions)))
    # bdd.dump("transition_constrant_expl_negated.pdf", roots=[~ transitions])


def build_symbolic_model(transition_sys_graph: TwoPlayerGame):
    """
    A helper function to create a symbolic graph
    """
    # bdd = BDD()
    # bdd.declare('x', 'y', 'z')
    # u = bdd.add_expr(r'x /\ y')
    #
    # values = dict(x=False, y=True)
    # v = bdd.let(values, u)
    # print(v)
    # # for d in bdd.pick_iter(u):
    # #     print(d)
    #
    # # for _ in range(1000):
    # #     d = bdd.pick(u, care_vars=['x', 'y'])
    # #     print(d)
    # bdd.collect_garbage()  # optional
    # bdd.dump('awesome.pdf')

    # sys.exit(-1)

    # relabel all nodes
    new_graph = transition_sys_graph.internal_node_mapping(transition_sys_graph.transition_system)

    # create a binary representation for each variable
    bdd = BDD()
    for _n in new_graph._graph.nodes():
        # _n = _n.replace(" ", "")
        new_str = "n" + str(_n)
        bdd.declare(new_str)

    # create a bdd for each transition. The expression is in TLA+ - a formal specification language
    trans_bdd_str_list = []
    counter = 0
    for _e in new_graph._graph.edges():
        # get the corresponding vars
        pre_node_var = bdd.level_of_var("n" + str(_e[0]))
        post_node_var = bdd.level_of_var("n" + str(_e[1]))

        # create bdd transition str
        tran_str = r'({u} & {v})'.format(u="n" + str(_e[0]),
                                         v="n" + str(_e[1]))
        trans_bdd_str_list.append(tran_str)

        # create a bdd for this transition
        # trans_bdd_list.append(bdd.add_expr(r'({u} & {v})'.format(u="n" + str(_e[0]),
        #                                                           v="n" + str(_e[1]))))
        #
        if counter == 1:
        #
        #     d = dict(n0=trans_bdd_list[1])
        #     f = bdd.let(d, trans_bdd_list[0])
        #     bdd.dump('transition.pdf')
        #
        #     for eval in bdd.pick_iter(f):
        #         print(eval)
        #
            break
        #
        counter += 1

    str_expr = ' | '.join(trans_bdd_str_list)
    print(str_expr)

    bdd.add_expr(str_expr)
    # bdd.collect_garbage()  # optional
    bdd.dump('transition_function.pdf')

    sys.exit(-1)


def get_graph(print_flag: bool = False):
    _project_root = os.path.dirname(os.path.abspath(__file__))

    _domain_file_path = _project_root + "/pddl_files/two_table_scenario/arch/domain.pddl"
    _problem_file_path = _project_root + "/pddl_files/two_table_scenario/arch/problem.pddl"

    _causal_graph_instance = CausalGraph(problem_file=_problem_file_path,
                                         domain_file=_domain_file_path,
                                         draw=False)

    _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

    if print_flag:
        print(
            f"No. of nodes in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.nodes())}")
        print(
            f"No. of edges in the Causal Graph is :{len(_causal_graph_instance._causal_graph._graph.edges())}")

    _transition_system_instance = FiniteTransitionSystem(_causal_graph_instance)
    _transition_system_instance.build_transition_system(plot=False, relabel_nodes=False)
    _transition_system_instance.build_arch_abstraction(plot=False, relabel_nodes=False)

    if print_flag:
        print(f"No. of nodes in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.nodes())}")
        print(f"No. of edges in the Transition System is :"
              f"{len(_transition_system_instance.transition_system._graph.edges())}")

    return _transition_system_instance


if __name__ == "__main__":
    # testing BDD theory
    # test_constraint_vs_explicit_transition_thr()
    # sys.exit(-1)

    # construct a sample rwo player game and wrap it to construct its symbolic version
    transition_graph = get_graph(print_flag=True)
    build_symbolic_model(transition_graph)