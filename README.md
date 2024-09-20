# NUCS

## TLDR
NUCS is a Python library for solving Constraint Satisfaction and Optimization Problems.
Because it is 100% written in Python, NUCS is easy to install and use.
NUCS is also very fast because it is powered by [Numpy](https://numpy.org/) and [Numba](https://numba.pydata.org/).

With NUCS, in a few seconds you can ...

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

## How to use NUCS ?
It is very simple to get started with NUCS.
You can either install the Pip package or install NUCS from the sources.

### Install the NUCS package
Let's install the Pip package for NUCS:
```bash
pip install nucs
````
Now we can write the following `queens.py` program (please refer to [the technical documentation](DOCUMENTATION.md) 
to better understand how NUCS works under the hood):
```python
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.propagators.propagators import ALG_ALLDIFFERENT

n = 8  # the number of queens
problem = Problem(
    [(0, n - 1)] * n,  # these n domains are shared between the 3n variables with different offsets
    list(range(n)) * 3,  # for each variable, its shared domain
    [0] * n + list(range(n)) + list(range(0, -n, -1))  # for each variable, its offset
)
problem.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
problem.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
problem.add_propagator((list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []))
print(BacktrackSolver(problem).solve_one()[:n])
```
Let's run this model with the following command:
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python queens.py
```
The first solution found is:
```bash
[0, 4, 7, 5, 2, 6, 1, 3]
```
> [!TIP]
> Note that the second run will always be __much faster__ since the Python code will already have been compiled and cached by Numba.

### Install NUCS from the sources 
Let's install NUCS from the sources by cloning the NUCS Github repository:
```bash
git clone https://github.com/yangeorget/nucs.git
pip install -r requirements.txt
```
From there, we will launch some NUCS examples.

#### Run some examples
Some of the examples come with a command line interface and can be run directly.

Let's find all solutions to the [12-queens problem](https://www.csplib.org/Problems/prob054/):
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_queens.py -n 12
```

Let's find the optimal solution to the [Golomb ruler problem](https://www.csplib.org/Problems/prob006/) with 10 marks:
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_golomb.py -n 10
```

## Other constraint solvers in Python
### CCPMpy
_CPMpy is a Constraint Programming and Modeling library in Python, based on numpy, with direct solver access._

### Numberjack
_Numberjack is a modelling package written in Python for combinatorial optimisation._

### python-constraint
_The Python constraint module offers solvers for Constraint Solving Problems (CSPs) over finite domains in simple and pure Python._

### PyCSP
_PyCSP3 is a Python library for developping models of combinatorial constrained problems in a declarative manner; you can write models of constraint satisfaction (CSP) and constraint optimization (COP) problems._