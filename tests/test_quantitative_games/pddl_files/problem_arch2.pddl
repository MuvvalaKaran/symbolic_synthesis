(define (problem arch_franka_world) (:domain dynamic_franka_box_world)
(:objects

    ;;;;; Locs where only the robot can operate ;;;;;
    
    l0 - hbox_loc
    l1 - hbox_loc
    l2 - hbox_loc

    ;;;;; Locs where the robot & human can operate ;;;;;
    ; NOTE: The way pyperplan parses the PDDL file, you need atleast two human locs to construct `human-move` action

    l6 - box_loc
    l7 - hbox_loc


    b0 - box
    b1 - box
    b2 - box
    


)


(:init
    (ready l0)
    
    (on b0 else)
    (on b1 else)
    (on b2 else)
)

(:goal 
(and
    (on b0 l0)
)

)

)