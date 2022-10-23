import re
import networkx as nx

from sympy import symbols

from ltlf2dfa.parser.ltlf import LTLfParser


class Ltlf2MonaDFA:
    """
    A class that call the LTLF2DFA python package and constructs DFA.

    Implementation: LTL2DFA package calls the Mona packages to construct a
     minimal DFA.
     
     Link: https://whitemech.github.io/LTLf2DFA/
     Link: https://www.brics.dk/mona/
      
     Syft is another toolbox that follows a symbolic approach. 
      1) It first converts the LTLf fomrula into a First-order Formula (FOF)
      2) Then, calls Mona to convert to Convert FOF to a minimal DFA.
      3) Finally, Syft uses BDD based symbolic representation to construct a 
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


    def parse_mona(self, mona_output):
        """
        Parse mona output and extract the initial, accpeting and other states. The edges are constructed
         by create_symbolic_ltlf_transition_system() method in SymbolicDFA() and SymbolicAddDFA() classes. 
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
    formula = 'F(a & F(b))'

    dfa_handle = Ltlf2MonaDFA(formula=formula)
