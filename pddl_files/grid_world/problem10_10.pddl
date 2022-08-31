(define (problem grid_one_agent)
	(:domain grid_world)
	
	(:objects skbn l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l90 l91 l92 l93 l94 l95 l96 l97 l98 l99 l100)
	
	(
	:init  
		   (agent skbn) 

		   ;;horizontal relationships
		   (leftOf l1 l2) (leftOf l2 l3) (leftOf l3 l4) (leftOf l4 l5) (leftof l5 l6) (leftOf l6 l7) (leftOf l7 l8) (leftOf l8 l9) (leftOf l9 l10)
		   (leftOf l11 l12) (leftOf l12 l13) (leftOf l13 l14) (leftOf l14 l15) (leftof l15 l16) (leftOf l16 l17) (leftOf l17 l18) (leftOf l18 l19) (leftOf l19 l20)
		   (leftOf l21 l22) (leftOf l22 l23) (leftOf l23 l24) (leftOf l24 l25) (leftof l25 l26) (leftOf l26 l27) (leftOf l27 l28) (leftOf l28 l29) (leftOf l29 l30)
		   (leftOf l31 l32) (leftOf l32 l33) (leftOf l33 l34) (leftOf l34 l35) (leftof l35 l36) (leftOf l36 l37) (leftOf l37 l38) (leftOf l38 l39) (leftOf l39 l40)
		   (leftOf l41 l42) (leftOf l42 l43) (leftOf l43 l44) (leftOf l44 l45) (leftof l45 l46) (leftOf l46 l47) (leftOf l47 l48) (leftOf l48 l49) (leftOf l49 l50)
		   (leftOf l51 l52) (leftOf l52 l53) (leftOf l53 l54) (leftOf l54 l55) (leftof l55 l56) (leftOf l56 l57) (leftOf l57 l58) (leftOf l58 l59) (leftOf l59 l60)
		   (leftOf l61 l62) (leftOf l62 l63) (leftOf l63 l64) (leftOf l64 l65) (leftof l65 l66) (leftOf l66 l67) (leftOf l67 l68) (leftOf l68 l69) (leftOf l69 l70)
		   (leftOf l71 l72) (leftOf l72 l73) (leftOf l73 l74) (leftOf l74 l75) (leftof l75 l76) (leftOf l76 l77) (leftOf l77 l78) (leftOf l78 l79) (leftOf l79 l80)
		   (leftOf l81 l82) (leftOf l82 l83) (leftOf l83 l84) (leftOf l84 l85) (leftof l85 l86) (leftOf l86 l87) (leftOf l87 l88) (leftOf l88 l89) (leftOf l89 l90)
		   (leftOf l91 l92) (leftOf l92 l93) (leftOf l93 l94) (leftOf l94 l95) (leftof l95 l96) (leftOf l96 l97) (leftOf l97 l98) (leftOf l98 l99) (leftOf l99 l100)

 		   ;;vertical relationships
 		   (below l1 l11) (below l2 l12) (below l3 l13) (below l4 l14) (below l5 l15) (below l6 l16) (below l7 l17) (below l8 l18) (below l9 l19) (below l10 l20)
 		   (below l11 l21) (below l12 l22) (below l13 l23) (below l14 l24) (below l15 l25) (below l16 l26) (below l17 l27) (below l18 l28) (below l19 l29) (below l20 l30)
 		   (below l21 l31) (below l22 l32) (below l23 l33) (below l24 l34) (below l25 l35) (below l26 l36) (below l27 l37) (below l28 l38) (below l29 l39) (below l30 l40)
 		   (below l31 l41) (below l32 l42) (below l33 l43) (below l34 l44) (below l35 l45) (below l36 l46) (below l37 l47) (below l38 l48) (below l39 l49) (below l40 l50)
 		   (below l41 l51) (below l42 l52) (below l43 l53) (below l44 l54) (below l45 l55) (below l46 l56) (below l47 l57) (below l48 l58) (below l49 l59) (below l50 l60)
 		   (below l51 l61) (below l52 l62) (below l53 l63) (below l54 l64) (below l55 l65) (below l56 l66) (below l57 l67) (below l58 l68) (below l59 l69) (below l60 l70)
 		   (below l61 l71) (below l62 l72) (below l63 l73) (below l64 l74) (below l65 l75) (below l66 l76) (below l67 l77) (below l68 l78) (below l69 l79) (below l70 l80)
 		   (below l71 l81) (below l72 l82) (below l73 l83) (below l74 l84) (below l75 l85) (below l76 l86) (below l77 l87) (below l78 l88) (below l79 l89) (below l80 l90)
 		   (below l81 l91) (below l82 l92) (below l83 l93) (below l84 l94) (below l85 l95) (below l86 l96) (below l87 l97) (below l88 l98) (below l89 l99) (below l90 l100)



 		   ;;initialize agent
		   (at skbn l1)
		   )

	(:goal (and (at skbn l100)

	))

)