import argparse

from typing import List, Union
from config import VAR_DICT, PROJECT_ROOT

ALGORITHM_CHOICES = ['regret', 'adv-game', 'gridworld', 'frankaworld']
TYPE_OF_GAME_CHOICES = ['quant-adv', 'quant-coop', 'qual']
TYPE_OF_TR = ['monolithic', 'partitioned']

def parse_args_base():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--domain",
                           default=PROJECT_ROOT + "/pddl_files/franka_regret_world/test/domain.pddl",
                           type=str, help="path to domain pddl file",)
    
    argparser.add_argument("--task",
                           default=PROJECT_ROOT + "/pddl_files/franka_regret_world/test/problem.pddl",
                           type=str, help="path to task pddl file")
    
    argparser.add_argument("--algorithm", type=str,
                           default='regret-min', choices=ALGORITHM_CHOICES,
                           help="Which algorithnm to use. Default: %(default)d")
    
    # argparser.add_argument("--type-of-game",
    #                        default='quant-adv', choices=TYPE_OF_GAME_CHOICES, type=str,
    #                        help="choose qual for qualitative game, quant-adv for quantitative adversarial game, and quant-coop for cooperative game. Default: %(default)d")

    # argparser.add_argument("--type-of-TR",
    #                        default='monolithic', choices=TYPE_OF_TR, type=str,
    #                        help="choose monolithic for monolithic TR and partitioned for partitioned TR. Default: %(default)d")
    
    # argparser.add_argument("--regret-hybrid",
    #                        action='store_true',
    #                        help="Set this flag to true when you want to contruct Graph of Utility and Graph of Best Response explicitly. Otherwise we constrcut it symbolically. Default: %(default)d")
    
    # argparser.add_argument("--no-regret-hybrid",
    #                        dest='regret_hybrid',
    #                        action='store_false')

    argparser.add_argument("--simulate",
                           action='store_true',
                           help="Rollout strategy. Default: %(default)d")
    
    argparser.add_argument("--no-simulate",
                           dest='simulate',
                           action='store_false')
    
    argparser.set_defaults(simulate=True)

    return argparser


def parse_args_regret(subparser):
    """
     A subparser that create arguments for the regret synthesis case
    """
    regret_argparser = subparser.add_parser('regret', help='Regret-minimizing strategy synthesis')


    regret_argparser.add_argument("--type-of-game",
                                  default='quant-adv', choices=TYPE_OF_GAME_CHOICES, type=str,
                                  help="choose qual for qualitative game, quant-adv for quantitative adversarial game, and quant-coop for cooperative game. Default: %(default)d")

    regret_argparser.add_argument("--type-of-TR",
                                  default='monolithic', choices=TYPE_OF_TR, type=str,
                                  help="choose monolithic for monolithic TR and partitioned for partitioned TR. Default: %(default)d")
    
    regret_argparser.add_argument("--regret-hybrid",
                                  action='store_true',
                                  help="Set this flag to true when you want to contruct Graph of Utility and Graph of Best Response explicitly. Otherwise we constrcut it symbolically. Default: %(default)d")
    
    regret_argparser.add_argument("--no-regret-hybrid",
                                  dest='regret_hybrid',
                                  action='store_false')
    
    regret_argparser.add_argument("--formula", type=str, required=True,
                                  help="Input Formula. We currentlt support LTLf and scLTL formulas")
    

    return regret_argparser.parse_args()


def parse_args_gridworld(subparser):
    """
     A subparser that create arguments for the regret synthesis case
    """
    gridworld_argparser = subparser.add_parser('gridworld', help='Regret-minimizing strategy synthesis')

    gridworld_argparser.add_argument("--formula", type=Union[List[str], str], required=True, 
                                     help="Input Formula(s). We currentlt support LTLf and scLTL formulas")
    
    return gridworld_argparser.parse_args()


def parse_args_frankaworld(subparser):
    """
     A subparser that create arguments for the Frankaworld case. Frankaworld is the
        Manipulator Domain without any human intervention.
    """
    frankaworld_argparser = subparser.add_parser('frankaworld', help='Regret-minimizing strategy synthesis')

    frankaworld_argparser.add_argument("--formula", type=Union[List[str], str], required=True, 
                                       help="Input Formula(s). We currentlt support LTLf and scLTL formulas")
    
    return frankaworld_argparser.parse_args()


def parse_args_only_adv(subparser):
    """
     A subparser that create arguments for the min-max game strategy synthesis. Here the
        Manipulator Domain has human intervention.
    
        Specifically: 
        1. TWO_PLAYER_GAME - this is for unlimited human intervention
        2. TWO_PLAYER_GAME_BND - this is for bounded human intervention - He et al. IROS17

    """
    adv_game_argparser = subparser.add_parser('adv-game', help='Regret-minimizing strategy synthesis')

    adv_game_argparser.add_argument("--formula", type=Union[List[str], str], required=True, 
                                    help="Input Formula(s). We currentlt support LTLf and scLTL formulas")
    
    return adv_game_argparser.parse_args()



def copy_args_to_dict(args):
    # update Var Dictionery file 
    VAR_DICT['domain'] = vars(args)['domain']
    VAR_DICT['problem'] = vars(args)['task']
    VAR_DICT['GAME_ALGORITHM'] = vars(args)['type_of_game']
    VAR_DICT['SIMULATE'] = True if vars(args)['simulate'] else False
    VAR_DICT['FORMULA'] = [vars(args)['formula']]
    VAR_DICT['MONOLITHIC_TR'] = True if vars(args)['type_of_TR'] == 'monolithic' else False
    VAR_DICT['REGRET_HYBRID'] = True if vars(args)['regret_hybrid'] else False



def setup():
    base_argparser = parse_args_base()
    subparsers = base_argparser.add_subparsers(help='sub-command help')
    args = base_argparser.parse_args()
    # initialize sub-argument based on the algo user chose
    if vars(args)['algorithm'] == 'regret':
        VAR_DICT['REGRET_SYNTHESIS'] = True
        updated_args = parse_args_regret(subparser=subparsers)
    elif vars(args)['algorithm'] == 'adv-game':
        VAR_DICT['STRATEGY_SYNTHESIS'] = True
        updated_args = parse_args_only_adv(subparser=subparsers)
    elif vars(args)['algorithm'] == 'frankaworld':
        VAR_DICT['FRANKAWORLD'] = True
        updated_args = parse_args_frankaworld(subparser=subparsers)
    elif vars(args)['algorithm'] == 'gridworld':
        VAR_DICT['GRIDWORLD'] = True
        updated_args = parse_args_gridworld(subparser=subparsers)

    copy_args_to_dict(updated_args)


if  __name__ == "__main__":
    print(VAR_DICT)