## Description

This directory contains test scripts to check if the source code for abstraction construction, strategy synthesis, and simulating the synthesized strategy/policy is functioning correctly or not. To run the test scripts, use the following command

```bash
cd <root/of/directory>

python3 -m tests.test_adversarial_game -b
```

The `-m` flag run the test as a module while `-b` flag suppresses the output of the code if all the tests pass. If the test fails, then it throw the Error message at the very begining and then the rest of the outputs.

### Directories

1. Quantitative_games: Contains Domain and problem file to construct 2 box 4 location Manipulator abstraction with unbounded human intervention. Box_loc are the locations where only the robot can operate and HBox_loc are locations where both the human and the robot can manipulate the objects placed.  