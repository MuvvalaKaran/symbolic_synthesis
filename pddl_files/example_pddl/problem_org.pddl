(define (problem simple_only_franka_world) (:domain simple_franka_box_world)
(:objects
    franka - robot
    else - robo_loc

    l0 - box_loc
    l1 - box_loc
    l2 - box_loc

    b0 - box
    b1 - box
    b2 - box

)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    
    (on b0 l0)
    (on b1 l1)
    (on b2 l2)
)

(:goal 
(and
    (on b0 l0)
    (on b1 l1)
    (on b2 l2)
)

)

)