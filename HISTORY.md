This file keep tracks of various version of this source code

### V2.1

- This tag implements a working Franka abstraction construction code using a mixture of conjuction (`&`) and disjusction (`|`) of boolean variables to represent each symbolic node in the graph.
	- I use `x` for boolean variables that represent robot status. For actions like `release` and `grasp` (domain - simple_franka_box_world), I take the union (`|`) of the boolean formulas of the form `(~x1 & ~x2) | (x1 & ~x2)` where `(~x1 & ~x2)` and `(x1 & ~x2)` could represent `(to-loc b0 l0)` and `(holding b0 l0)` as preconditions for `release` motion primitive (Note: `~` is the negation operator).
	- I use `b` for boolean variables that represent current world configuration. So, a world conf. where `(on b0 l0)`, `(on b1 l1)`, and `(gripper free)`  is true, I represent it as `(~b1 & ~b2) | (b1 & ~b2) | (~b1 & b2)`. 
	- Thus, a symbolic state can be fully defined as the conjunction of `((~x1 & ~x2) | (x1 & ~x2)) & ((~b1 & ~b2) | (b1 & ~b2) | (~b1 & b2))`. While this representation requires few boolean variables as compared to a monolithic representation (the whole state is expressed using one boolean variable say `x`), the algorithms like `image` and `pre-image` have to be modified to accomodate the `|` operator. 

### V2.0

- This tag implements a Grid World abstraction construction in a Monolithic Fashion.
	- We use two sets of boolean variables to explicitly represent current and next states in the TS and DFA Transition Relation.
- This version also implements LTL and LTLf synthesis code. We call Mona to conpute a minimal DFA and parse its output to construct a symbolic DFA.
- Finally, this version implements A\*, Dijkstras, and BFS graph search algorithms using the monolithic representation of the composed TR.


### V1.0

- This tag corresponds to BFS, Dijkstra search algorithms for single and multiple DFAs. 
- Details about the search algorithm 
	- We create two different types of variables for TS and DFA 
	- We search over the symbolic graphs in a disjointed fashion, compute image on TS, check the evolution on the DFA/DFAs one at time
	- update the search frontier, repeat.
	- This approach is inherently slower due to multiple image calls in one iteration and multuple calls to extract cubes
