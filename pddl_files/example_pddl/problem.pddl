(define (problem simple_only_franka_world) (:domain simple_franka_box_world)
(:objects
    franka - robot
    else - robo_loc

    l0 - box_loc
    l1 - box_loc
    l2 - box_loc
    l3 - box_loc
    l4 - box_loc
    ;l5 - box_loc
    ;l6 - box_loc
    ;l7 - box_loc
    ;l8 - box_loc
    ;l9 - box_loc
    ;l10 - box_loc

    b0 - box
    b1 - box
    b2 - box
    b3 - box


)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    (gripper free)
    
    (on b0 l0)
    (on b1 l1)
    (on b2 l2)
    (on b3 l3)
)

(:goal 
(and
    (on b0 l0)
)

)

)