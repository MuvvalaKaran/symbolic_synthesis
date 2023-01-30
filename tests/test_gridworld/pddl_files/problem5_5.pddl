(define (problem grid_one_agent)
	(:domain grid_world)
	
	(:objects skbn l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 l21 l22 l23 l24 l25)
	(
	:init  
		   (agent skbn) 

		   ;;horizontal relationships
		   (leftOf l1 l2) (leftOf l2 l3) (leftOf l3 l4) (leftOf l4 l5)
		   (leftOf l6 l7) (leftOf l7 l8) (leftOf l8 l9) (leftOf l9 l10)
		   (leftOf l11 l12) (leftOf l12 l13) (leftOf l13 l14) (leftOf l14 l15) 
		   (leftOf l16 l17) (leftOf l17 l18) (leftOf l18 l19) (leftOf l19 l20)
		   (leftOf l21 l22) (leftOf l22 l23) (leftOf l23 l24) (leftOf l24 l25) 

 		   ;;vertical relationships
 		   (below l1 l6) (below l2 l7) (below l3 l8) (below l4 l9) (below l5 l10)
 		   (below l6 l11) (below l7 l12) (below l8 l13) (below l9 l14) (below l10 l15)
 		   (below l11 l16) (below l12 l17) (below l13 l18) (below l14 l19) (below l15 l20)
 		   (below l16 l21) (below l17 l22) (below l18 l23) (below l19 l24) (below l20 l25)


 		   ;;initialize agent
		   (at skbn l1)
		   )

	(:goal (and (at skbn l25)

	))
)