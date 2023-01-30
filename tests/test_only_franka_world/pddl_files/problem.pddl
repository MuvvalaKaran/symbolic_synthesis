(define (problem simple_only_franka_world) (:domain simple_franka_box_world)
(:objects
    franka - robot
    else - robo_loc

    l0 - box_loc
    l1 - box_loc
    l2 - box_loc
    
    l6 - box_loc
    l7 - box_loc
    
    b0 - box
    b1 - box
    b2 - box

)


(:init
    (ready else)
    (gripper free)
    
    (on b0 l0)
    (on b1 l6)
    (on b2 l7)
)

(:goal 
(and
    (on b0 l0)
)

)

)