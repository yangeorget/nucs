![pypi version](https://img.shields.io/pypi/v/nucs?color=blue&label=pypi%20version&logo=pypi&logoColor=white)
![numba version](https://img.shields.io/badge/numba-v0.60-blue)
![numpy version](https://img.shields.io/badge/numpy-v2.0-blue)
![tests](https://github.com/yangeorget/nucs/actions/workflows/test.yml/badge.svg)
![doc](https://img.shields.io/readthedocs/nucs)
![license](https://img.shields.io/github/license/yangeorget/nucs)

## TLDR
NUCS is a Python library for solving Constraint Satisfaction and Optimization Problems.
Because it is 100% written in Python, NUCS is easy to install and use.
NUCS is also very fast because it is powered by [Numpy](https://numpy.org/) and [Numba](https://numba.pydata.org/).

## Documentation
Check out [NUCS documentation](https://nucs.readthedocs.io/).

## With NUCS, in a few seconds you can ...
Compute the 92 solutions to the [BIBD(8,14,7,4,3) problem](https://www.csplib.org/Problems/prob028/):
```python
{
    'OPTIMIZER_SOLUTION_NB': 0,
    'PROBLEM_FILTER_NB': 2797,
    'PROBLEM_PROPAGATOR_NB': 462,
    'PROBLEM_VARIABLE_NB': 504,
    'PROPAGATOR_ENTAILMENT_NB': 36977,
    'PROPAGATOR_FILTER_NB': 564122,
    'PROPAGATOR_FILTER_NO_CHANGE_NB': 534436,
    'PROPAGATOR_INCONSISTENCY_NB': 1307,
    'SOLVER_BACKTRACK_NB': 1398,
    'SOLVER_CHOICE_DEPTH': 41,
    'SOLVER_CHOICE_NB': 1398,
    'SOLVER_SOLUTION_NB': 92
}
```
Demonstrate that the optimal [10-marks Golomb ruler](https://www.csplib.org/Problems/prob006/) length is 55:
```python
{
    'OPTIMIZER_SOLUTION_NB': 10,
    'PROBLEM_FILTER_NB': 22204,
    'PROBLEM_PROPAGATOR_NB': 82,
    'PROBLEM_VARIABLE_NB': 45,
    'PROPAGATOR_ENTAILMENT_NB': 416934,
    'PROPAGATOR_FILTER_NB': 2145268,
    'PROPAGATOR_FILTER_NO_CHANGE_NB': 1129818,
    'PROPAGATOR_INCONSISTENCY_NB': 11065,
    'SOLVER_BACKTRACK_NB': 11064,
    'SOLVER_CHOICE_DEPTH': 9,
    'SOLVER_CHOICE_NB': 11129,
    'SOLVER_SOLUTION_NB': 10
 }
```
Find all 14200 solutions to the [12-queens problem](https://www.csplib.org/Problems/prob054/):
```python
{
    'OPTIMIZER_SOLUTION_NB': 0,
    'PROBLEM_FILTER_NB': 262011,
    'PROBLEM_PROPAGATOR_NB': 3,
    'PROBLEM_VARIABLE_NB': 36,
    'PROPAGATOR_ENTAILMENT_NB': 0,
    'PROPAGATOR_FILTER_NB': 1910609,
    'PROPAGATOR_FILTER_NO_CHANGE_NB': 631079,
    'PROPAGATOR_INCONSISTENCY_NB': 116806,
    'SOLVER_BACKTRACK_NB': 131005,
    'SOLVER_CHOICE_DEPTH': 10,
    'SOLVER_CHOICE_NB': 131005,
    'SOLVER_SOLUTION_NB': 14200
}
```
