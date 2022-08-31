(define (problem grid_one_agent)
	(:domain grid_world)
	
	(:objects skbn l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l30 l31 l32 l33 l34 l35 l36)
	(
	:init  
		   (agent skbn) 

		   ;;horizontal relationships
		   (leftOf l1 l2) (leftOf l2 l3) (leftOf l3 l4) (leftOf l4 l5) (leftof l5 l6)
		   (leftOf l7 l8) (leftOf l8 l9) (leftOf l9 l10) (leftOf l10 l11) (leftof l11 l12)
		   (leftOf l13 l14) (leftOf l14 l15) (leftOf l15 l16) (leftOf l16 l17) (leftof l17 l18)
		   (leftOf l19 l20) (leftOf l20 l21) (leftOf l21 l22) (leftOf l22 l23) (leftof l23 l24)
		   (leftOf l25 l26) (leftOf l26 l27) (leftOf l27 l28) (leftOf l28 l29) (leftof l29 l30)
		   (leftOf l31 l32) (leftOf l32 l33) (leftOf l33 l34) (leftOf l34 l35) (leftof l35 l36)


 		   ;;vertical relationships
 		   (below l1 l7) (below l2 l8) (below l3 l9) (below l4 l10) (below l5 l11) (below l6 l12)
 		   (below l7 l13) (below l8 l14) (below l9 l15) (below l10 l16) (below l11 l17) (below l12 l18)
 		   (below l13 l19) (below l14 l20) (below l15 l21) (below l16 l22) (below l17 l23) (below l18 l24)
 		   (below l19 l25) (below l20 l26) (below l21 l27) (below l22 l28) (below l23 l29) (below l24 l30)
 		   (below l25 l31) (below l26 l32) (below l27 l33) (below l28 l34) (below l29 l35) (below l30 l36)


 		   ;;initialize agent
		   (at skbn l1)
		   )

	(:goal (and (at skbn l36)

	))

)