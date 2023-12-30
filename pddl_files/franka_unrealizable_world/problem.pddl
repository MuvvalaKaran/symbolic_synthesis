(define (problem dynamic_unrealizable_franka_world) (:domain dynamic_unrealizable_franka_box_world)
(:objects
    
    ;;;;; In this problem domain, human can interven from all locations and hence no winning strategy exists. 
    ;;;;; Locs where the robot & human can operate ;;;;;
    ; NOTE: The way pyperplan parses the PDDL file, you need atleast two human locs to construct `human-move` action


    
    l0 - hbox_loc
    l1 - hbox_loc
    l2 - hbox_loc
    l3 - hbox_loc
    l4 - hbox_loc
    l5 - hbox_loc
    l6 - hbox_loc
    l7 - hbox_loc
    l8 - hbox_loc
    l9 - hbox_loc
    ;l10 - hbox_loc
    ;l11 - hbox_loc

    b0 - box
    b1 - box
    ;b2 - box
    ;b3 - box
    ;b4 - box
    ;b5 - box
    ;b6 - box


)

;todo: put the initial state's facts and numeric values here
(:init
    (ready else)
    
    (on b0 l0)
    (on b1 l6)
    ;(on b2 l7)
    ;(on b3 l1)
    ;(on b4 l3)
    ;(on b5 l5)
    ;(on b6 l6)
)

(:goal 
(and
    (on b0 l0)
)

)

)