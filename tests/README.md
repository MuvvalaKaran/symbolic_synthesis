## Description

This directory contains test scripts to check if the source code for abstraction construction, strategy synthesis, and simulating the synthesized strategy/policy is functioning correctly or not. To run the test scripts, use the following command to run one module

```bash
cd <root/of/project>

python3 -m tests.test_adversarial_game -b
```

The `-m` flag runs the test as a module while `-b` flag suppresses the output of the code if all the tests pass. If the test fails, then it throws the error message at the very begining and then the rest of the output. 

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