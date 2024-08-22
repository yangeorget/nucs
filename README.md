# NUCS

## TLDR
NUCS is a Python library for solving Constraint Satisfaction and Optimization Problems.
NUCS is powered by Numpy (https://numpy.org/) and Numba (https://numba.pydata.org/).
NUCS is fast and easy to use.


## How to use NUCS ?
It is very simple to get started with NUCS.
Either clone the Github repository or install the Pip package.

### Clone the NUCS Github repository
Let's install NUCS from the source:
```bash
git clone https://github.com/yangeorget/nucs.git
pip install -r requirements.txt
```

From there, you can launch some NUCS examples.  

> Note that the second run will be much faster since the Python code will have been compiled.

#### Run the NUCS tests
```bash
pip install -r requirements-dev.txt
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/
```

#### Run some examples
Find all solutions to the 12-queens problem:
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python3 tests/examples/test_queens.py -n 12
```

Find the optimal solution to the Golomb ruler problem with 10 marks:
```bash
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python3 tests/examples/test_golomb.py -n 10
```

### Install the NUCS package
```bash
pip3 install nucs
````
Now you can write the following queens.py program:
```python
from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.propagators.propagators import ALG_ALLDIFFERENT

n = 8
problem = Problem(shr_domains=[(0, n - 1)] * n, dom_indices=list(range(n)) * 3, dom_offsets=[0] * n + list(range(n)) + list(range(0, -n, -1)))
problem.set_propagators([(list(range(n)), ALG_ALLDIFFERENT, []), (list(range(n, 2 * n)), ALG_ALLDIFFERENT, []), (list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, [])])
print(BacktrackSolver(problem).solve_one())
```
Run it with the command:
```bash
PYTHONPATH=. python3 queens.py
```
You will get:
```bash
[0, 4, 7, 5, 2, 6, 1, 3, 0, 5, 9, 8, 6, 11, 7, 10, 0, 3, 5, 2, -2, 1, -5, -4]
```

## Why Python ?
NUCS is a Python library leveraging Numpy and Numba.

Python is a powerful and flexible programing language that allows to express complex problems in a few lines of code.

Numpy brings the computational power of languages like C and Fortran to Python, a language much easier to learn and use.

Numba translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. 
Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.

## Architecture

## Speed matters

## Other constraint solvers in Python
- python-constraint 
- CCPMpy
- PyCSP