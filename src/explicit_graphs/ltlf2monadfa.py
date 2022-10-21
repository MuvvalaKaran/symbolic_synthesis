import re
import networkx as nx

from typing import Union
from sympy import And, Not, Or, simplify, symbols

from ltlf2dfa.parser.ltlf import LTLfParser


class Ltlf2MonaDFA:
    """
    A class that call the LTLF2DFA python package and constructs DFA.
    """

    def __init__(self, formula: str, verbose: bool = False, plot: bool = False):
        self.formula: str = formula
        self._graph = nx.MultiDiGraph(name='ltlf_graph')
        self.task_labels: list = []
        self.init_state: list = []
        self.accp_states: list = []
        self.num_of_states = None
        self.mona_dfa: str = ''

        # Construct mona DFA
        self.construct_dfa(verbose=verbose, plot=plot)
    

    def get_value(self, text, regex, value_type=float):
        """
        Dump a value from a file based on a regex passed in.
        """
        pattern = re.compile(regex, re.MULTILINE)
        results = pattern.search(text)
        if results:
            return value_type(results.group(1))
        else:
            print("Could not find the value {}, in the text provided".format(regex))
            return value_type(0.0)


    def ter2symb(self, ap, ternary):
        """
        Translate ternary output to symbolic.
        """
        expr = And()
        i = 0
        for value in ternary:
            if value == "1":
                expr = And(expr, ap[i] if isinstance(ap, tuple) else ap)
            elif value == "0":
                assert value == "0"
                expr = And(expr, Not(ap[i] if isinstance(ap, tuple) else ap))
            else:
                assert value == "X", "[ERROR]: the guard is not X"
            i += 1
        return expr


    def simplify_guard(self, guards):
        """
        Make a big OR among guards and simplify them.
        """
        return simplify(Or(*guards))    


    def parse_mona(self, mona_output):
        """
        Parse mona output and construct a dot.
        """
        free_variables = self.get_value(
            mona_output, r".*DFA for formula with free variables:[\s]*(.*?)\n.*", str
        )
        if "state" in free_variables:
            free_variables = None
        else:
            free_variables = symbols(
                " ".join(
                    x.strip().lower() for x in free_variables.split() if len(x.strip()) > 0
                )
            )
        
        # store task specific labels
        self.task_labels = free_variables

        self.init_state = [1]
        accepting_states = self.get_value(mona_output, r".*Accepting states:[\s]*(.*?)\n.*", str)
        accepting_states = [
            str(x.strip()) for x in accepting_states.split() if len(x.strip()) > 0
        ]
        # store accepting states
        self.accp_states = [int(i) for i in accepting_states]

        # store # DFA states
        self.num_of_states = self.get_value(mona_output, '.*Automaton has[\s]*(\d+)[\s]states.*', int) - 1

        # create graph
        self._graph.add_nodes_from([s for s in range(1, self.num_of_states + 1)])

        # dot_trans = dict()  # maps each couple (src, dst) to a list of guards
        # for line in mona_output.splitlines():
        #     if line.startswith("State "):
        #         orig_state = self.get_value(line, r".*State[\s]*(\d+):\s.*", int)
        #         guard = self.get_value(line, r".*:[\s](.*?)[\s]->.*", str)
        #         if free_variables:
        #             guard = self.ter2symb(free_variables, guard)
        #         else:
        #             guard = self.ter2symb(free_variables, "X")
        #         dest_state = self.get_value(line, r".*state[\s]*(\d+)[\s]*.*", int)
        #         if orig_state:
        #             if (orig_state, dest_state) in dot_trans.keys():
        #                 dot_trans[(orig_state, dest_state)].append(guard)
        #             else:
        #                 dot_trans[(orig_state, dest_state)] = [guard]
        
        # self.transition = dot_trans

    #     for c, guards in dot_trans.items():
    #         simplified_guard = self.__simplify_guard(guards)
    #         dot += ' {} -> {} [label="{}"];\n'.format(
    #             c[0], c[1], str(simplified_guard).lower()
    #         )

    #     dot += "}"
    #     return dot
    

    def construct_dfa(self,
                      verbose: bool = False,
                      plot: bool = False):
        """
        A helper function that calls Mona and then parse the output and construct a DFA.
        """
        parser = LTLfParser()
        formula = parser(self.formula)       # returns an LTLf Formula

        # LTLf to Mona DFA
        mona_dfa = formula.to_dfa(mona_dfa_out=True)
        
        self.mona_dfa = mona_dfa
        
        if verbose:
            print("********************Mona DFA********************")
            print(mona_dfa)  

        
        if "Formula is unsatisfiable" in mona_dfa:
            print("Unsat Formula")
        else:
            my_dfa = self.parse_mona(mona_dfa)
        
        return my_dfa


if __name__ == "__main__":
    # formulas 
    formula = 'F(a & F(b))'

    dfa_handle = Ltlf2MonaDFA(formula=formula)
    dfa_handle.construct_dfa()
