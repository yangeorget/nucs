# NCS

## TLDR
NCS is a Python library for solving Constraint Satisfaction and Optimization Problems.

NCS is powered by Numpy (https://numpy.org/) and Numba (https://numba.pydata.org/).

NCS is fast and easy to use.


## How to use NCS ?
It is very simple to get started with NCS.
Either clone the Github repository or install the package.

### Clone the NCS Github repository
Let's install NCS from the source:
```
git clone https://github.com/yangeorget/ncs.git
pip install -r requirements.txt
```

From there, you can launch some NCS examples.
Note that the second run will be much faster since the Python code will have been compiled.

#### Run the NCS tests
```
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/
```

#### Run some examples
Find all solutions to the 12-queens problem:
```
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python3 tests/examples/test_queens.py -n 12
```

Find the optimal solution to the Golomb ruler problem with 10 marks:
```
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python3 tests/examples/test_golomb.py -n 10
```

### Install the NCS package
```
pip install ncs
````

## Why Python ?
NCS is a Python library leveraging Numpy and Numba.

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