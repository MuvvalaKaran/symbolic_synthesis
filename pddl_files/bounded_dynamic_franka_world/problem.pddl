(define (problem bounded_dynamic_only_franka_world) (:domain bounded_dynamic_franka_box_world)
(:objects
    franka - robot
    else - robo_loc

    ;;;;; Locs where only the robot can operate ;;;;;
    l0 - hbox_loc
    l1 - hbox_loc
    l2 - hbox_loc
    ;l3 - hbox_loc
    ;l4 - hbox_loc
    ;l5 - hbox_loc

    ;;;;; Locs where the robot & human can operate ;;;;;
    ; NOTE: The way pyperplan parses the PDDL file, you need atleast two human locs to construuct `human-move` action

    l6 - hbox_loc
    l7 - hbox_loc
    ;l8 - hbox_loc
    ;l9 - hbox_loc
    ;l10 - hbox_loc

    b0 - box
    b1 - box
    b2 - box
    ;b3 - box
    ;b4 - box
    ;b5 - box
    ;b6 - box


)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    (gripper free)
    
    (on b0 l0)
    (on b1 l6)
    (on b2 l2)
    ;(on b3 l3)
    ;(on b4 l6)
    ;(on b5 l5)
    ;(on b6 l6)
)

(:goal 
(and
    (on b0 l0)
)

)

)