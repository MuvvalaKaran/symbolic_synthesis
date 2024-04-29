# Directories

## Simple_franka_world

Contains all the files relevant to the Manipulator Domain without any human intervention.

1. domain.pddl and problem.pddl - Defines the domain for manipulator case-study and problem instance respectively.
2. domain_org.pddl and problem_org.pddl: Defines the manipulator domain from the original regret synthesis toolbox code for explicit Transition System construction. 

Note: The preconditions and Effects in the domain.pddl have been updated to streamline the symbolc abstraction construction process.

## Dynamic_franka_world

Manipulator Domain with human interventions

1. domain.pddl - Added an additional `type - hbox_loc`; a location where both the human and the robot can manipulate and move objects. Furthermore, Added `human-move` action to incorporate human actions.

## Grid_world

Contains all the files relevant to the 2D gridworld Agent.

1. domain.pddl: The domain file with cardinal actions - moveLeft, moveRight, moveUp, and moveDown for the gridworld agent (skbn -  sokoban for short).
2. problemN_N.pddl: Instance of the domain files. N represents the Gridworld size. We also have pddl file with a L shaped obstacle in a 20 by 20 gridworld.   


## FRANKA Regret world

Contains all the files relevant to the Regret Minimizing experiments for a dynamic franka world, i.e, with human intervention. The semantics of the domain file is the same as the described in `Dynamic_franka_world` section. 

## IROS23

Contains all the files relevant to the Regret Minimizing  IROS23 experiments (benchmarking +_ ARIA LAB construction) for a dynamic franka world, i.e, with human intervention. The semantics of the domain file is the same as the described in `Dynamic_franka_world` section. 