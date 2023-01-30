(define (domain grid_world)
	(:requirements :strips)

	(:predicates (agent ?x)   								    ;agent is at location x
				 (leftOf ?x ?y) 								;location x is to the left of locaiton y
				 (below ?x ?y)  								;location x is below location y
				 (at ?x ?y)                                     ;object x is at location y
				 )

	(:action moveLeft
		:parameters (?skbn ?x ?y)
		:precondition (and (agent ?skbn)
						   (at ?skbn ?x)
						   (leftOf ?y ?x))   					;location y is to the left of location x
		:effect (and (at ?skbn ?y)
				(not (at ?skbn ?x)))
				)

	(:action moveRight
		:parameters (?skbn ?x ?y)
		:precondition (and (agent ?skbn)
							(at ?skbn ?x)
							(leftOf ?x ?y))    					;location x is to the left of y
		:effect (and (at ?skbn ?y) 
				(not (at ?skbn ?x)))
				)

	(:action moveUp
		:parameters (?skbn ?x ?y)
		:precondition (and (agent ?skbn)
						  (at ?skbn ?x)
						  (below ?x ?y))      					;location x is below location y
		:effect (and (at ?skbn ?y) 
				(not (at ?skbn ?x)))
				)

	(:action moveDown
		:parameters (?skbn ?x ?y)
		:precondition (and (agent ?skbn)
						  (at ?skbn ?x)
						  (below ?y ?x))    					;location y is below location x
		:effect (and (at ?skbn ?y)
				(not (at ?skbn ?x)))
				)
)