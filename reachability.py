from dd import autoref as _bdd


def transition_system(bdd):
    """Return the transition relation of a graph."""
    dvars = ["x0", "x0'", "x1", "x1'"]
    for var in dvars:
        bdd.add_var(var)
    s = r'''
           ((~ x0 /\ ~ x1) => ( (~ x0' /\ ~ x1') \/ (x0' /\ ~ x1') ))
        /\ ((x0 /\ ~ x1) => ~ (x0' /\ x1'))
        /\ ((~ x0 /\ x1) => ( (~ x0' /\ x1') \/ (x0' /\ ~ x1') ))
        /\ ~ (x0 /\ x1)
        '''
    transitions = bdd.add_expr(s)

    return transitions


def least_fixpoint(transitions, bdd):
    """Return ancestor nodes.
    
    Detailed explanation of Algo.

    while no reached fixed-point:
        1. Start with expr that corresponds to the target set => q = target at the end of 1st iteration
        2. Check edges that can reach the target set => transitions & next_q
    
    
    
    """

    # target is the set {2}
    target = bdd.add_expr(r'~ x0 /\ x1')
    # start from empty set
    q = bdd.false
    qold = None
    prime = {"x0": "x0'", "x1": "x1'"}
    qvars = {"x0'", "x1'"}
    # fixpoint reached ?
    counter  = 0
    while q != qold:
        # print("**************************************************************************")
        print(f"****************************Counter: {counter} *********************************")
        qold = q
        print("q: ", q.to_expr())
        print(list(bdd.pick_iter(q)))
        next_q = bdd.let(prime, q)
        print("next_q: ", q.to_expr())
        print(list(bdd.pick_iter(next_q)))
        u = transitions & next_q   # compute the edges that lead to the target set
        print("u: ", u.to_expr())
        print(list(bdd.pick_iter(u)))
        # existential quantification over x0', x1'
        pred = bdd.exist(qvars, u)  # check the predecessor state given the edges computed
        print("pred", pred.to_expr())
        print(list(bdd.pick_iter(pred)))
        # alternative: pred = bdd.quantify(u, qvars, forall=False)
        q = q | pred | target
        print("q: ", q.to_expr())
        print(list(bdd.pick_iter(q)))
        counter += 1
        # print("**************************************************************************")
    return q


def forward_reachability(transitions, bdd):
    """
    Return set of all reachable states form the initial state
    """

    # init set is {0}
    # init_set = bdd.add_expr(r'~ x0 /\ ~ x1')

    # init set is {0, 1}
    # init_set = bdd.add_expr(r'~ x1')

    # init set is all the states
    init_set = bdd.true

    # start from empty set
    q = bdd.false
    qold = None
    prime = {"x0": "x0'", "x1": "x1'"}
    qvars = {"x0'", "x1'"}
    qvars_forward = {"x0", "x1"}
    prime_forward = {"x0'": "x0", "x1'": "x1"}

    # fixpoint reached ?
    counter = 0
    while q != qold:
        print(f"****************************Counter: {counter} *********************************")
        qold = q
        print("q: ", q.to_expr())
        print(list(bdd.pick_iter(q)))
        u = transitions & q  # compute the edges that lead from the reachable set
        print("u: ", u.to_expr())
        print(list(bdd.pick_iter(u)))
        # existential quantification over x0, x1
        succ = bdd.exist(qvars_forward, u)  # check the successor states given the edges computed
        print("succ", succ.to_expr())
        print(list(bdd.pick_iter(succ)))
        # alternative: pred = bdd.quantify(u, qvars, forall=False)
        next_succ = bdd.let(prime_forward, succ)
        print("next_succ: ", next_succ.to_expr())
        print(list(bdd.pick_iter(next_succ)))
        q = q | init_set | next_succ
        print("q: ", q.to_expr())
        print(list(bdd.pick_iter(q)))
        counter += 1
    return q


def depth_first_search(transitions, bdd):
    """
    Return sets of states after you find A valid path to the target set
    """

    # init set is {0}
    init_set = bdd.add_expr(r'~ x0 /\ ~ x1')

    # target set is {2}
    target = bdd.add_expr(r'~x0 /\ x1')

    # start from empty set
    q = bdd.false
    qvars_forward = {"x0", "x1"}
    prime_forward = {"x0'": "x0", "x1'": "x1"}

    # initialize flag to be false
    check = bdd.false

    # breadth first algorithm
    counter = 0

    # THE exprs check is non trivial if the intersection of reachable states & target set is nonzero
    while check.to_expr() == 'FALSE':
        print(f"****************************Counter: {counter} *********************************")
        u = transitions & q  # compute the edges that lead from the reachable set
        # existential quantification over x0, x1
        succ = bdd.exist(qvars_forward, u)  # check the successor states given the edges computed
        # as successor states are in terms of primes, convert prime -> unprime
        next_succ = bdd.let(prime_forward, succ)

        # take union of the reachable set, successor states computed in this loop & init set
        q = q | init_set | next_succ

        # check if any state in target set has been reached
        check = q & target
        # print(list(bdd.pick_iter(check)))
        counter += 1

    print(f"Reached Target state in {counter} steps")


if __name__ == '__main__':
    bdd = _bdd.BDD()
    transitions = transition_system(bdd)
    # q = least_fixpoint(transitions, bdd)
    q = forward_reachability(transitions, bdd)

    depth_first_search(transitions, bdd)
    # s = q.to_expr()
    # print(s)