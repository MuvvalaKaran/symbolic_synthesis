This file keep tracks of various version of this source code

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
