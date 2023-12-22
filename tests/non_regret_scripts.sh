#!/bin/bash

# Initialize verbose to false
VERBOSE=false

# Check all arguments
for arg in "$@"
do
  if [ "$arg" == "-v" ] || [ "$arg" == "--verbose" ]; then
    VERBOSE=true
  fi
done

# Check if the first argument is -h or --help
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Usage: '$0' [pass -v or --verbose for verbose test results]"
  exit 0
fi

if $VERBOSE; then
  echo "Verbose mode is on."

  python3 -m unittest discover -s tests.test_dfa_construction -bv
  python3 -m unittest discover -s tests.test_gridworld -bv
  python3 -m unittest discover -s tests.test_only_franka_world -bv
  python3 -m unittest discover -s tests.test_qualitative_games -bv
  python3 -m unittest discover -s tests.test_quantitative_games -bv
  python3 -m unittest discover -s tests.test_bnd_human_games -bv
else
  python3 -m unittest discover -s tests.test_dfa_construction -b
  python3 -m unittest discover -s tests.test_gridworld -b
  python3 -m unittest discover -s tests.test_only_franka_world -b
  python3 -m unittest discover -s tests.test_qualitative_games -b
  python3 -m unittest discover -s tests.test_quantitative_games -b
  python3 -m unittest discover -s tests.test_bnd_human_games -b

fi