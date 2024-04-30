import argparse
import sys

from config import VAR_DICT, PROJECT_ROOT

ALGORITHM_CHOICES = ['regret-min', 'min-max', 'gridworld', 'frankaworld']
TYPE_OF_GAME_CHOICES = ['regret-min', 'min-max', 'gridworld', 'frankaworld']
TYPE_OF_TR = ['monolithic', 'partitioned']

def parse_args():
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
    
    argparser.add_argument("--type-of-game",
                           default='quant-adv', choices=TYPE_OF_GAME_CHOICES, type=str,
                           help="choose qual for qualitative game, quant-adv for quantitative adversarial game, and quant-coop for cooperative game. Default: %(default)d")

    argparser.add_argument("--type-of-TR",
                           default='monolithic', choices=TYPE_OF_TR, type=str,
                           help="choose monolithic for monolithic TR and partitioned for partitioned TR. Default: %(default)d")
    
    argparser.add_argument("--regret-hybrid",
                           action='store_true',
                           help="Set this flag to true when you want to contruct Graph of Utility and Graph of Best Response explicitly. Otherwise we constrcut it symbolically. Default: %(default)d")
    
    argparser.add_argument("--no-regret-hybrid",
                           dest='regret_hybrid',
                           action='store_false')

    argparser.add_argument("--simulate",
                           action='store_true',
                           help="Rollout strategy. Default: %(default)d")
    
    argparser.add_argument("--no-simulate",
                           dest='simulate',
                           action='store_false')
    
    argparser.add_argument("--formula", type=str, required=True,
                           help="Input Formula. We currentlt support LTLf and scLTL formulas")
    
    argparser.set_defaults(simulate=True)

    return argparser.parse_args()


def copy_args_to_dict(args):
    # update Var Dictionery file 
    VAR_DICT['domain_file_path'] = vars(args)['domain']
    VAR_DICT['problem_file_path'] = vars(args)['task']

    # for _ in range(len(ALGORITHM_CHOICES)):
    if vars(args)['algorithm'] == 'regret-min':
        VAR_DICT['REGRET_SYNTHESIS'] = True
        VAR_DICT['STRATEGY_SYNTHESIS'] = False
        VAR_DICT['FRANKAWORLD'] = False
        VAR_DICT['GRIDWORLD'] = False
    
    VAR_DICT['GAME_ALGORITHM'] = vars(args)['type_of_game']
    VAR_DICT['SIMULATE'] = True if vars(args)['simulate'] else False
    VAR_DICT['FORMULA'] = [vars(args)['formula']]
    VAR_DICT['MONOLITHIC_TR'] = True if vars(args)['type_of_TR'] == 'monolithic' else False
    VAR_DICT['REGRET_HYBRID'] = True if vars(args)['regret_hybrid'] else False



def setup():
    args = parse_args()
    copy_args_to_dict(args)


if  __name__ == "__main__":
    print(VAR_DICT)