# Project Description

This repository contains code for the paper "Efficient Symbolic Approaches for Quantitative Reactive Synthesis with Finite Tasks." In this work we introduce symbolic algorithms for quantitative reactive synthesis. We consider resource constrained robotic manipulators that need to interact with a human to achieve a complex task expressed in linear temporal logic. This code implement Symbolic Value Iteration for Min-max Reachbility games with quantative constraints as well Hybrid Regret-Minimizing strategy synthesis algorithm. See the [paper](https://arxiv.org/abs/2303.03686) for more details. 

# Authors

* [Karan Muvvala](https://muvvalakaran.github.io/) <[karan.muvvala@colorado.edu](mailto:karan.muvvala@colorado.edu)>

## Instructions to clone the code

clone this repo with:
 ```bash
git clone --recurse-submodules git@github.com:aria-systems-group/sym_quant_reactive_synth.git .
```

Make sure you have Docker installed. If not, follow the instructions [here](https://docs.docker.com/engine/install/ubuntu/).


Note: The ` --recurse-submodule` will automatically initialize and update each submodule in the repository. **The command will throw an error if you do not have access to the private submodule `cudd_and_wrapper`. Please send me an email at :karan.muvvala@colorado.edu so that I can give access to the dependent submodule.**


## Docker Commands to build the image

1. `cd` into the root of the project

2. Build the image from the Dockerfile

```bash
docker build -t <image_name> .
```

Note: the dot looks for a Dockerfile in the current repository.Then spin an instance of the container by using the following command

```bash
docker run -it --name <docker_container_name> <docker image name>

```

For volume binding

```bash
docker run -v <HOST-PATH>:<Container-path>
```

Additionally, if you are more used to GUI and would like to edit or attach a container instance to the VSCode ([Link](https://code.visualstudio.com/docs/devcontainers/containers)) then follow the instructions below:

### Attaching the remote container to VScode


1. Make sure you have the right VS code extensions installed
	* install docker extension
	* install python extension
	* install remote container extension
	* Now click on the `Remote Explore` tab on the left and attach VScode to a container.
2. This will launch a new vs code attached to the container and prompt you to a location to attach to. The default is root, and you can just press enter. Congrats, you have attached your container to VSCode.

## Tests

All the tests related scripts are available in the `tests/` directory. I use python [unittest](https://docs.python.org/3.8/library/unittest.html) for testing individual components of my source code. Here are some commands to run the tests:

To run a specific test package:

```bash
python3 -m unittest discover -s tests.<directory-name> -bv
```

To run a specific test script:

```bash
python3 -m tests.<directory-name>.<module-nane> -b
```

To run all tests:

```bash
python3 -m unittest -bv
```

For more details see the `tests/README.md`. Note, all commands must be run from `<root/of/project>`. Additionally, I have included to shell scripts to run regret and non-regret unit tests. You can call these <test_script>.sh file with `-v` (default is False) for verbose test results.

## Directories


* src/symbolic_graphs - Contains code for running the main scripts and constructing the graphs
* src/algorithms - contains various synthesis code. 
* test - test scripts to check if the code for abstraction construction, strategy synthesis, and simulating the synthesized strategy/policy is functioning correctly.
* pddl_files - PDDL domains for static and dynamic manipulator domain. Also contains Grid world examples.


### Submodules

* Cudd_and_wrapper - A cython-based wrapper for the CUDD decision diagram package.
* Regret_synthesis_toolbox - Source code for the explicit implementatiob of Min-max reachability algorithm and Regret-minimizing strategy synthesis. See this [paper](https://arxiv.org/pdf/2203.06861.pdf) for more information.


## Benchmarking Dataset

The PDDL files used for benchmarking the symbolic synthesis code (Fig. 5 from Experiments section) can be found [here](https://github.com/MuvvalaKaran/conference-datasets).

## Contributing
All contributions welcome! All content in this repository is licensed under the MIT license.

## Citing

If the code is useful in your research, and you would like to acknowledge it, please cite this [paper](https://arxiv.org/abs/2303.03686):

```
@article{muvvala2023efficient,
  title={Efficient Symbolic Approaches for Quantitative Reactive Synthesis with Finite Tasks},
  author={Muvvala, Karan and Lahijanian, Morteza},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  year={2023},
  pages={8666-8672},
  doi={10.1109/IROS55552.2023.10342496}}
}
```

# Contact

Please contact me if you have questions at :karan.muvvala@colorado.edu