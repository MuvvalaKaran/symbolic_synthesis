This file keep tracks of various versions of this source code

### V6.0

This tag implements working symbolic Regret Synthesis Code. Added code to construct Graph of Utility and Graph of Best Response. Implementation Details:

1. hybrid_graph_of_utility - This script includes code to construct the graph of utility. Refer to [this](https://muvvalakaran.github.io/publication/#7) for definition and algorithm. For this tag, we explicitly roll out the original graph to construct the Symbolic TR. The bottleneck of this approach is the Computation time.

2. hybrid_game_abstraction - Similar to Graph of Utility, we explicitly loop over each edge to compute the best alternate response and then proceed to construct the symbolic TR for Regret-minimizing strategy synthesis by playing a Min-Max game over this new graph. This implementation also suffers from Computation time. 


### V5.0

This tag implements symbolic strategy synthesis for Two-player zero-sum games with quantitative constraints under adversarial and cooperative human assumptions. The abstraction is similar to the Unbounded Abstraction. The graph evolves under system and human action from a given state in the Transition System. Each edge also has an edge weight associated with. This, to represent edge weights we use Algebraic Decision Diagrams (ADDs). We make use of the Compositional Approach along with ADDs for strategy synthesis for the Robot player. The task is defined using LTLf language.

Adversarial (Min-Max) Game: The Robot player and the Human player have competing objectives. We assume the human to be adversarial, i.e., the objective for the system is to minimize the total cost (Cumulative Payoff) to accomplish the task while the objective is to maximize the total energy the Robot player needs to spend to accomplish the same task. Thus, we call System/Robot player as Min player and Human/Environment player as Max player as well. We are playing a Zero-sum game thus the payoff for the Robot player is exactly the opposite of the payoff of the Human player. Thus, we also call this a Min-Max game.

Note: The Reactive synthesis community also calls the Human player as Input and Robot player as the Output player. The Robot chooses an action (Output) for all possible Human actions (inputs). Thus, in my code I represent Human actions as `I` and Robot actions as `O`. 

Cooperative (Min-Min) Game: The Robot player and the Human player both have the same objective, i.e., both players want to the minimize the total energy spent by the Robot player to accomplish the given task.  Thus, we also call this a Min-Min game.

### V4.0

This tag implements a faster and smaller abstraction construction and quicker winning strategy synthesis code.

Major improvements - 
 1. Updated Domain file for Bounded and Unbounded Abstraction - fewer predicates leading to fewer states in TS and thus fewer boolean variables.  
 2. Instead of creating all possible configurations, we break the state encoding into Robot Conf and World Conf expressed using `X`and B<sub>i</sub> boolean variables. 
 3. Modified Robot and Human actions. 
 4. The variables are created in the order `I`,  `O`, `Z`, B<sub>i</sub>, `X`, `K`. 
 5. In the strategy synthesis, we first evolve over DFA variables `Z`, then compute the pre of the TS states. This eliminates the need for "Hook" and thus remove state label issue. 

Bounded Abstraction: The abstraction is similar to the one used by Keliang. From each state in the TS, you have a robot edge and a human edge. You either evolve according to human edge or robot edge. During roll out, I toss a coin and choose for either the human to intervene or not intervene (evolve according to Robot's action).

Unbounded Abstraction: The above abstraction interpretation does not work due to the unbounded nature of human interventions. For e.g., at a given TS state say the robot wants to `release` the box in End Effector but the human keeps moving some other block. Using the above interpretation, the graph will always evolve as per the human move and thus the robot won't be able to `release` the box. 

To resolve this, in this abstraction, the graph evolves as per the combined robot and human action, i.e., (TS-state) ---- (release) & (human move) ---> (TS-state)'. Here (TS-state)' reflects robot's intention to perform an action (release) and the actual evolution as per the human's action (move some other block). Refer to [Commit](https://github.com/MuvvalaKaran/symbolic_synthesis/commit/0684e4f5e18aa6dbc86dbb929bbda2d27617057c) for more details.

### State Encoding

### V3.0

This Tag implements a slow winning strategy synthesis code for Bounded and Unbounded Franka abstraction.
	
* ### State Encoding

	In this implementation, each state in the TS is composed of the following Boolean variables - `I`; `O`; B<sub>i</sub>; `G`; `X`; `K`;`Z`. Here, 

		1. `I` - Human Action Vars
		2. `O` - Robot Action vars
		3. `X` - State Vars
		4. `B_i` - Propositional Vars for each box 
		5. `G` - Gripper status Var
		6. `Z`- DFA state vars
		7. `K` - Vars that keep track of # of human actions remaining
	
	We precompute all the possible Configurations (Robot + World Conf.), assign each state a boolean formula (made up of X vars), create TR in a compositional fashion ([link](https://drive.google.com/file/d/1UUW-HgJ_CgiMFufWic1C5Z842nRMuaUZ/view?usp=sharing)). Each edge in TS includes variables (I & O & B<sub>i</sub> & G & X & K & Z) in that specific variable order.

* ### Strategy Synthesis 

We use the state labels expressed as Boolean formulas made up of B<sub>i</sub> & G as "hooks". During the predecessor computation, the pre of the winning states is computed. The labels of the current states are used as hooks to evolve on the DFA, i.e., to update DFA variables `Z`. Due to this "hook" based implementation, after every predecessor computation, we need to update the state labels of each state to be the correct one. This issue is resolved in the next implementation.

### V2.1
 This tag implements a working Franka abstraction construction code using a mixture of conjunction (`&`) and disjunction (`|`) of boolean variables to represent each symbolic node in the graph.
- I use `x` for boolean variables that represent robot status. For actions like `release` and `grasp` (domain - simple_franka_box_world), I take the union (`|`) of the boolean formulas of the form `(~x1 & ~x2) | (x1 & ~x2)` where `(~x1 & ~x2)` and `(x1 & ~x2)` could represent `(to-loc b0 l0)` and `(holding b0 l0)` as preconditions for `release` motion primitive (Note: `~` is the negation operator).
- I use `b` for boolean variables that represent current world configuration. So, a world conf. where `(on b0 l0)`, `(on b1 l1)`, and `(gripper free)`  is true, I represent it as `(~b1 & ~b2) | (b1 & ~b2) | (~b1 & b2)`. 
- Thus, a symbolic state can be fully defined as the conjunction of `((~x1 & ~x2) | (x1 & ~x2)) & ((~b1 & ~b2) | (b1 & ~b2) | (~b1 & b2))`. While this representation requires few boolean variables as compared to a monolithic representation (the whole state is expressed using one boolean variable say `x`), the algorithms like `image` and `pre-image` have to be modified to accomodate the `|` operator. 

### V2.0

- This tag implements a Grid World abstraction construction in a Monolithic Fashion.
	- We use two sets of boolean variables to explicitly represent current and next states in the TS and DFA Transition Relation.
- This version also implements LTL and LTLf synthesis code. We call Mona to compute a minimal DFA and parse its output to construct a symbolic DFA.
- Finally, this version implements A\*, Dijkstra's, and BFS graph search algorithms using the monolithic representation of the composed TR.


### V1.0

- This tag corresponds to BFS, Dijkstra search algorithms for single and multiple DFAs. 
- Details about the search algorithm 
	- We create two different types of variables for TS and DFA 
	- We search over the symbolic graphs in a disjointed fashion, compute image on TS, check the evolution on the DFA/DFAs one at time
	- update the search frontier, repeat.
	- This approach is inherently slower due to multiple image calls in one iteration and multiple calls to extract cubes
