import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPLICIT_GRAPH: bool = False  # set this flag to true when you want to construct Explicit graph

QUANTITATIVE_SEARCH: bool = False  # set this flag to true whe you have edge costs

BUILD_DFA: bool = True
BUILD_ABSTRACTION: bool = True
CREATE_VAR_LBLS: bool = True   # set this to true if you want to create Observation BDDs

DRAW_EXPLICIT_CAUSAL_GRAPH: bool = False
SIMULATE_STRATEGY: bool = False
GRID_WORLD_SIZE: int = 5
OBSTACLE: bool = False  # galf to load the onbstacle gridworl and color the gridworld accordingly