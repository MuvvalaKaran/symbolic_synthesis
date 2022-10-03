This file keep tracks of various version of this source code

### V1.0

- This tag corresponds to BFS, Dijkstra search algorithms for single and multiple DFAs. 
- Details about the search algorithm 
	- We create two different types of variables for TS and DFA 
	- We search over the symbolic graphs in a disjointed fashion, compute image on TS, check the evolution on the DFA/DFAs one at time
	- update the search frontier, repeat.
	- This approach is inherently slower due to multiple image calls in one iteration and multuple calls to extract cubes
