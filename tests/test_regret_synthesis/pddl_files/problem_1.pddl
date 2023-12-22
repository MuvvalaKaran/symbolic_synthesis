(define (problem dynamic_regret_franka_world) (:domain dynamic_regret_franka_box_world)
(:objects
    ;;;; Test that checks for correct BA set computation. See Gh-Issue #3 for more info.
    ;;;;; Locs where only the robot can operate ;;;;;
    
    l0 - box_loc
    l1 - box_loc

    ;;;;; Locs where the robot & human can operate ;;;;;
    ; NOTE: The way pyperplan parses the PDDL file, you need atleast two human locs to construct `human-move` action

    l6 - hbox_loc
    l7 - hbox_loc

    b0 - box
    b1 - box

)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    
    (on b0 l0)
    (on b1 l6)
)

(:goal 
(and
    (on b0 l0)
))

)