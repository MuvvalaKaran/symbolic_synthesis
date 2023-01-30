## Description

This directory contains test scripts to check if the source code for abstraction construction, strategy synthesis, and simulating the synthesized strategy/policy is functioning correctly or not. 

To run each test package, use the following command

```bash
python3 -m unittest discover -s tests.<directory-name> -bv
```

The `-s` flag allows you to specify directory to start discovery from. Use only `-b` if you want to suppress all the prints (including progress). Use `-v` for verbose print or `-bv` to just print if a test failed or pass.   

To run the test scripts within each package, use the following command to run one module

```bash
cd <root/of/project>

python3 -m tests.<directory-name><module-nane> -b
```

The `-m` flag runs the test as a module while `-b` flag suppresses the output of the code if all the tests pass (you can also use `-bv`). If the test fails, then it throws the error message at the very begining and then the rest of the output. 


To run all the tests use the following command

```bash
python3 -m unittest -bv
```

### Directories

1. Quantitative_games: Contains Domain and problem file to construct 2 box 4 location Manipulator abstraction with unbounded human intervention. Box_loc are the locations where only the robot can operate and HBox_loc are locations where both the human and the robot can manipulate the objects placed.
	1. problem: Problem related to simple manipulation
	2. problem_arch1: Constructing arch with all object's desired location out of human's reach.
	3. problem_arch2: Constucting arch with all object's desired location within human's reach. We get different strategy for `quant-adv` and `quant-coop` strategy synthesis. 

2. Qualitative_games: Tests all the implementations related to abstraction construction and winning strategy synthesis for the manipulation domain. 

3. Gridworld: Tests strategy synthesis for a robot operating in a 2d gridworld with cardinal actions. The tests verify synthesis for single formula and multiple forumals (LTL and LTLf) with (dijkstras, A*) and without edge weights.

4. DFA_construction: This file tests the LTL/ LTLf to DFA construction. We test three type DFA construction for
	1. SymboliDFA() - A class used to create the DFA for the 2d gridworld exmaples using BDD Variables. The formulas can be LTL or LTLf!
    2. SymbolicDFAFranka() - A class used to create DFA for the Manipulation examples in MONOLITHIC Fashion. The formulas can only be LTLf!
    3. PartitionedDFA() - A class used to create DFA for the Manipulation example in COMPOSITIONAL Fashion. The formulas can only be LTLf!