(define (problem grid_one_agent)
	(:domain grid_world)
	
	(:objects skbn l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16)
	(
	:init  
		   (agent skbn) 

		   ;;horizontal relationships
		   (leftOf l1 l2) (leftOf l2 l3) (leftOf l3 l4)
		   (leftOf l5 l6) (leftOf l6 l7) (leftOf l7 l8)
		   (leftOf l9 l10) (leftOf l10 l11) (leftOf l11 l12)  
		   (leftOf l13 l14) (leftOf l14 l15) (leftOf l15 l16)

 		   ;;vertical relationships
 		   (below l1 l5) (below l2 l6) (below l3 l7) (below l4 l8)
 		   (below l5 l9) (below l6 l10) (below l7 l11) (below l8 l12)
 		   (below l9 l13) (below l10 l14) (below l11 l15) (below l12 l16)


 		   ;;initialize agent
		   (at skbn l1)
		   )

	(:goal (and (at skbn l16)

	))
)