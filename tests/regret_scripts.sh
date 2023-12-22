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
  python3 -m tests.test_regret_synthesis.test_hybrid_monolithic_reg_str_synth -bv
  python3 -m tests.test_regret_synthesis.test_hybrid_reg_str_synth_one_chance -bv
  python3 -m tests.test_regret_synthesis.test_hybrid_reg_str_synth_two_chance -bv
  python3 -m tests.test_regret_synthesis.test_hybrid_monolithic_reg_str_synth_issue_3 -bv
  python3 -m tests.test_regret_synthesis.test_symbolic_monolithic_reg_str_synth -bv
  python3 -m tests.test_regret_synthesis.test_symbolic_reg_str_synth_one_chance -bv
  python3 -m tests.test_regret_synthesis.test_symbolic_reg_str_synth_two_chance -bv
  python3 -m tests.test_regret_synthesis.test_symbolic_monolithic_reg_str_synth_issue_3 -bv
else
  python3 -m tests.test_regret_synthesis.test_hybrid_monolithic_reg_str_synth -b
  python3 -m tests.test_regret_synthesis.test_hybrid_reg_str_synth_one_chance -b
  python3 -m tests.test_regret_synthesis.test_hybrid_reg_str_synth_two_chance -b
  python3 -m tests.test_regret_synthesis.test_hybrid_monolithic_reg_str_synth_issue_3 -b
  python3 -m tests.test_regret_synthesis.test_symbolic_monolithic_reg_str_synth -b
  python3 -m tests.test_regret_synthesis.test_symbolic_reg_str_synth_one_chance -b
  python3 -m tests.test_regret_synthesis.test_symbolic_reg_str_synth_two_chance -b
  python3 -m tests.test_regret_synthesis.test_symbolic_monolithic_reg_str_synth_issue_3 -b
fi

