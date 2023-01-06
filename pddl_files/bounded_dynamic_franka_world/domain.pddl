;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Two-player Franka Blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (domain bounded_dynamic_franka_box_world)

(:requirements :strips :typing)

(:types
    box - object
    location - object
    general_loc - location
    ee_loc - location
    box_loc - general_loc
    hbox_loc - box_loc
)

; Indicates box is in end effector
(:constants else - general_loc ee - ee_loc) 

(:predicates
    ;(holding ?b - box ?l - general_loc)
    (holding ?l - general_loc)

    ;(ready ?o - object)
    (ready ?l - general_loc)

    ;(to-obj ?b - box ?l - general_loc)
    (to-obj ?b - box)

    ;(to-loc ?b - box ?l - box_loc)
    (to-loc ?l - box_loc)

    (on ?b - box ?l - location)
    
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; define actions here
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(:action transit
    :parameters (?b - box ?l1 - general_loc ?l2 - general_loc)
    :precondition (and 
        (ready ?l1)
        (on ?b ?l2)
    )
    :effect (and 
        (to-obj ?b)
        ;(ready ?b)
        (not (ready ?l1))
    )
)


(:action grasp
    :parameters (?b - box ?l - general_loc)
    :precondition (and 
        (to-obj ?b)
        ;(ready ?b)
        (on ?b ?l)
    )
    :effect (and 
        (holding ?l)
        ;(ready ?l)
        (on ?b ee)
        ;(not (ready ?l))
        (not (to-obj ?b))
        (not (on ?b ?l))
    )
)

(:action transfer
    :parameters (?b - box ?l1 - general_loc ?l2 - box_loc)
    :precondition (and 
        (holding ?l1)
        ;(ready ?l1)
        (on ?b ee)
    )
    :effect (and 
        (to-loc ?l2)
        ;(ready ?l2)
        (on ?b ee)
        ;(not (ready ?l1))
        (not (holding ?l1))

    )
)

(:action release
    :parameters (?b - box ?l - box_loc)
    :precondition (and
        (to-loc ?l)
        ;(ready ?l)
        (on ?b ee)
    )
    :effect (and
        (ready ?l)
        (on ?b ?l)
        (not (to-loc ?l))
        (not (on ?b ee))
    )
)


(:action human-move
    :parameters (?b - box ?l1 - hbox_loc ?l2 - hbox_loc)
    :precondition (and
        (on ?b ?l1)
    )
    :effect (and
        (on ?b ?l2)
        (not (on ?b ?l1))
    )
)


)