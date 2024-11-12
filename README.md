![NucS logo](https://github.com/yangeorget/nucs/blob/main/assets/nucs.png?raw=true)


![pypi version](https://img.shields.io/pypi/v/nucs?color=blue&label=pypi%20version&logo=pypi&logoColor=white)
![pypi downloads](https://img.shields.io/pypi/dm/NUCS)

![numba version](https://img.shields.io/badge/numba-v0.60-blue)
![numpy version](https://img.shields.io/badge/numpy-v2.0-blue)

![tests](https://github.com/yangeorget/nucs/actions/workflows/test.yml/badge.svg)
![doc](https://img.shields.io/readthedocs/nucs)
![license](https://img.shields.io/github/license/yangeorget/nucs)

## TLDR
NuCS is a Python library for solving Constraint Satisfaction and Optimization Problems.
Because it is 100% written in Python, 
NuCS is easy to install and allows to model complex problems in a few lines of code.
The NuCS solver is also very fast because it is powered by [Numpy](https://numpy.org/) and [Numba](https://numba.pydata.org/).

## Installation
```bash
pip install nucs
```
## Documentation
Check out [NUCS documentation](https://nucs.readthedocs.io/).

## With NuCS, in a few seconds you can ...
### Find all 14200 solutions to the [12-queens problem](https://www.csplib.org/Problems/prob054/)
```bash
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 12 --log_level=INFO
```
```bash
2024-11-12 17:24:49,061 - INFO - nucs.solvers.solver - Problem has 3 propagators
2024-11-12 17:24:49,061 - INFO - nucs.solvers.solver - Problem has 12 variables
2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses variable heuristic 0
2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses domain heuristic 0
2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses consistency algorithm 0
2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - Choice points stack has a maximal height of 128
2024-11-12 17:24:49,200 - INFO - nucs.solvers.multiprocessing_solver - MultiprocessingSolver has 1 processors
{
    'ALG_BC_NB': 262011,
    'ALG_BC_WITH_SHAVING_NB': 0,
    'ALG_SHAVING_NB': 0,
    'ALG_SHAVING_CHANGE_NB': 0,
    'ALG_SHAVING_NO_CHANGE_NB': 0,
    'PROPAGATOR_ENTAILMENT_NB': 0,
    'PROPAGATOR_FILTER_NB': 2269980,
    'PROPAGATOR_FILTER_NO_CHANGE_NB': 990450,
    'PROPAGATOR_INCONSISTENCY_NB': 116806,
    'SOLVER_BACKTRACK_NB': 131005,
    'SOLVER_CHOICE_NB': 131005,
    'SOLVER_CHOICE_DEPTH': 10,
    'SOLVER_SOLUTION_NB': 14200
}
```

### Compute the 92 solutions to the [BIBD(8,14,7,4,3) problem](https://www.csplib.org/Problems/prob028/)
```bash
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3 --symmetry_breaking --log_level=INFO
```
```bash
2024-11-12 17:26:39,734 - INFO - nucs.solvers.solver - Problem has 462 propagators
2024-11-12 17:26:39,734 - INFO - nucs.solvers.solver - Problem has 504 variables
2024-11-12 17:26:39,734 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses variable heuristic 0
2024-11-12 17:26:39,734 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses domain heuristic 1
2024-11-12 17:26:39,734 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses consistency algorithm 0
2024-11-12 17:26:39,734 - INFO - nucs.solvers.backtrack_solver - Choice points stack has a maximal height of 128
{
    'ALG_BC_NB': 1425,
    'ALG_BC_WITH_SHAVING_NB': 0,
    'ALG_SHAVING_NB': 0,
    'ALG_SHAVING_CHANGE_NB': 0,
    'ALG_SHAVING_NO_CHANGE_NB': 0,
    'PROPAGATOR_ENTAILMENT_NB': 4711,
    'PROPAGATOR_FILTER_NB': 104392,
    'PROPAGATOR_FILTER_NO_CHANGE_NB': 73792,
    'PROPAGATOR_INCONSISTENCY_NB': 621,
    'SOLVER_BACKTRACK_NB': 712,
    'SOLVER_CHOICE_NB': 712,
    'SOLVER_CHOICE_DEPTH': 19,
    'SOLVER_SOLUTION_NB': 92
}
```

### Demonstrate that the optimal [10-marks Golomb ruler](https://www.csplib.org/Problems/prob006/) length is 55
```bash
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 10 --symmetry_breaking --log_level=INFO
```
```bash
2024-11-12 17:27:45,110 - INFO - nucs.solvers.solver - Problem has 82 propagators
2024-11-12 17:27:45,110 - INFO - nucs.solvers.solver - Problem has 45 variables
2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses variable heuristic 0
2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses domain heuristic 0
2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses consistency algorithm 2
2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - Choice points stack has a maximal height of 128
2024-11-12 17:27:45,172 - INFO - nucs.solvers.backtrack_solver - Minimizing variable 8
2024-11-12 17:27:45,644 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 80
2024-11-12 17:27:45,677 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 75
2024-11-12 17:27:45,677 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 73
2024-11-12 17:27:45,678 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 72
2024-11-12 17:27:45,679 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 70
2024-11-12 17:27:45,682 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 68
2024-11-12 17:27:45,687 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 66
2024-11-12 17:27:45,693 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 62
2024-11-12 17:27:45,717 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 60
2024-11-12 17:27:45,977 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 55
{
    'ALG_BC_NB': 22652,
    'ALG_BC_WITH_SHAVING_NB': 0,
    'ALG_SHAVING_NB': 0,
    'ALG_SHAVING_CHANGE_NB': 0,
    'ALG_SHAVING_NO_CHANGE_NB': 0,
    'PROPAGATOR_ENTAILMENT_NB': 107911,
    'PROPAGATOR_FILTER_NB': 2813035,
    'PROPAGATOR_FILTER_NO_CHANGE_NB': 1745836,
    'PROPAGATOR_INCONSISTENCY_NB': 11289,
    'SOLVER_BACKTRACK_NB': 11288,
    'SOLVER_CHOICE_NB': 11353,
    'SOLVER_CHOICE_DEPTH': 9,
    'SOLVER_SOLUTION_NB': 10
}
[ 1  6 10 23 26 34 41 53 55]
```


