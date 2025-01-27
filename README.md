![NucS logo](https://raw.githubusercontent.com/yangeorget/nucs/main/assets/nucs.png)


![pypi version](https://img.shields.io/pypi/v/nucs?color=blue&label=pypi%20version&logo=pypi&logoColor=white)
![pypi downloads](https://img.shields.io/pypi/dm/NUCS)

![python version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fyangeorget%2Fnucs%2Fmain%2Fpyproject.toml)
![numba version](https://img.shields.io/badge/numba-v0.61-blue)
![numpy version](https://img.shields.io/badge/numpy-v2.1.3-blue)

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
Check out [NuCS documentation](https://nucs.readthedocs.io/).

## With NuCS, in a few seconds you can ...
### Find all 14200 solutions to the [12-queens problem](https://www.csplib.org/Problems/prob054/)
```bash
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 12
```
![queens](https://raw.githubusercontent.com/yangeorget/nucs/main/assets/queens.gif)

### Compute the 92 solutions to the [BIBD(8,14,7,4,3) problem](https://www.csplib.org/Problems/prob028/)
```bash
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3
```
![bibd](https://raw.githubusercontent.com/yangeorget/nucs/main/assets/bibd.gif)

### Demonstrate that the optimal [10-marks Golomb ruler](https://www.csplib.org/Problems/prob006/) length is 55
```bash
NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 10
```
![golomb](https://raw.githubusercontent.com/yangeorget/nucs/main/assets/golomb.gif)


