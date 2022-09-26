import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPLICIT_GRAPH: bool = False  # set this flag to true when you want to construct Explicit graph

QUANTITATIVE_SEARCH: bool = False  # set this flag to true when you have edge costs

BUILD_DFA: bool = True
BUILD_ABSTRACTION: bool = True
CREATE_VAR_LBLS: bool = True   # set this to true if you want to create Observation BDDs

DRAW_EXPLICIT_CAUSAL_GRAPH: bool = False
SIMULATE_STRATEGY: bool = True
GRID_WORLD_SIZE: int = 5
OBSTACLE: bool = False  # galf to load the onbstacle gridworl and color the gridworld accordingly
PRINT_STRATEGY: bool = False
DYNAMIC_VAR_ORDERING: bool = True


 # 5 by 5 formulas
# formulas = ["F(l21) & F(l5) & F(l25) & F(l1)",
#             "F(l19) & F(l7) & F(l9) & F(l17)",
#             "F(l23) & F(l3) & F(l11) & F(l15)",
#             "F(l16) & F(l24) & F(l2) & F(l10) "
#             ]

# formulas = ["F(l21) & F(l5) & F(l25) & F(l1)",
#             "F(l22) & F(l4) & F(l20) & F(l6)",
#             "F(l23) & F(l3) & F(l11) & F(l15)",
#             "F(l16) & F(l24) & F(l2) & F(l10) "
#             ]

# formulas = ['F(l2)',
#             'F(l91)', 
#             'F(l93)',
#             'F(l4)', 
#             'F(l95)',
#             'F(l6)',
#             'F(l97)',
#             'F(l8)',
#             'F(l99)',
#             'F(l10)'
#             ] 




# for 20 by 20 grid world
# formulas = ["F(l191 & F(l110) & F(l200))",
#             "F(l289 & F(l212) & F(l119))",
#             "F(l123 & F(l13) & F(l111))",
#             "F(l165 & F(l324) & F(l32))"
#             ]
# formulas = ['F(l7)', 'F(l13)', 'F(l19)', 'F(l25)']
# formulas = ['F(l7)', 'F(l13)', 'F(l19)']
# formulas = ['F(l13)', 'F(l7)']
# formulas = ['F(l25)', 'F(l2)', 'F(l21)', 'F(l5)']

# 5 state formula for 5x5 GW
formulas = ["F(l21 & F(l5) & F(l25))",
            "F(l22 & F(l4) & F(l20))",
            "F(l23 & F(l3) & F(l15))",
            "F(l24 & F(l2) & F(l10))",
            "F(l16 & F(l21) & F(l2))",
            "F(l11 & F(l22) & F(l3))",
            "F(l6 & F(l23) & F(l4))",
            "F(l2 & F(l20) & F(l16))",
            ]



# list of formula
# formulas = [
#     'F(l25)',
#     '!l2 & !l7 U l13',
#     'F(l25) & F(l15)',
#     'F(l19 & F(l13))',   # simple Formula w 2 states
#     # 'F(l13 & (F(l21) & F(l5)))',
#     # 'F(l6) & F(l2)', 
#     # 'F(l13 & (F(l21 & (F(l5)))))',
#     # "F(l21 & (F(l5 & (F(l25 & F(l1))))))",   # traversing the gridworld on the corners
#     # "F(l91 & (F(l10 & (F(l100 & F(l1))))))"   # traversing the gridworld on the corners for 10 x 10 gridworld
#     # "F(l400)",
#     # "F(l100 & F(l1))",
#     # "F(l100 & F(l1 & F(l91)))"
#     # "F(l381 & (F(l20 & (F(l400 & F(l1))))))",   # traversing the gridworld on the corners for 20 x 20 gridworld
#     # "F(l381 & (F(l20 & (F(l400)))))",
#     # "F(l381 & (F(l20)))",
#     ]