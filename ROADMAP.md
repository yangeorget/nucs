# Constraints
- GCC
- Cycle
- Product

# Problems
- submit problems to CSPlib
- RoundRobin problem (GCC)
- Job-shop problem 
- Golfers problem
- Car sequencing problem (GCC or Element)
- Crypto multiplication problem (Product)
- review CSPlib for additional problem

# Engine
- choice points pruning
- backtrackable propagator __state__
- implement decision variables

# Numba
- use prange with parallel=True
- test ufunc
- test vectorize

# Docs
- fix README
- remove install from source
- improve reference with problems and examples:  NUMBA_CACHE_DIR=.numba/cache PYTHON_PATH=. python -m nucs.examples.golomb.golomb_problem -n 10
- improve usage of rst
