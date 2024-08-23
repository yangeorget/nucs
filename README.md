# NUCS

## TLDR
NUCS is a Python library for solving Constraint Satisfaction and Optimization Problems.
NUCS is powered by [Numpy](https://numpy.org/) and [Numba](https://numba.pydata.org/).
NUCS is fast and easy to use.

## Speed matters
TODO: give some examples of computations in a limited time

## How to use NUCS ?
It is very simple to get started with NUCS.
You can either install the Pip package or install NUCS from the sources.

### Install the NUCS package
Let's install the Pip package for NUCS:
```bash
pip3 install nucs
````
Now we can write the following `queens.py` program, 
refer to [the technical documentation](DOCUMENTATION.md) to better understand how NUCS works under the hood:
```python
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.propagators.propagators import ALG_ALLDIFFERENT

n = 8  # the number of queens
problem = Problem(
    shr_domains=[(0, n - 1)] * n,  # these n domains are shared between 3n variables with different offsets
    dom_indices=list(range(n)) * 3,  # fpr each variable, its domain
    dom_offsets=[0] * n + list(range(n)) + list(range(0, -n, -1))  # for each variable, its offset
)
problem.set_propagators([
    (list(range(n)), ALG_ALLDIFFERENT, []), 
    (list(range(n, 2 * n)), ALG_ALLDIFFERENT, []), 
    (list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, [])
])
print(BacktrackSolver(problem).solve_one()[:n])
```
Let's run this model with the following command:
```bash
PYTHONPATH=. python3 queens.py
```
The first solution found is:
```bash
[0, 4, 7, 5, 2, 6, 1, 3]
```
> Note that the second run will always be much faster since the Python code will have been compiled by Numba.

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
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python3 tests/examples/test_queens.py -n 12
```

Let's find the optimal solution to the [Golomb ruler problem](https://www.csplib.org/Problems/prob006/) with 10 marks:
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python3 tests/examples/test_golomb.py -n 10
```

## Other constraint solvers in Python
- python-constraint 
- CCPMpy
- PyCSP