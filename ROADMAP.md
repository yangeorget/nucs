# Constraints
- GCC
- Cycle
- Product
- Alldifferent: relire http://www2.ift.ulaval.ca/~quimper/publications/ijcai03_TR.pdf et ecrire tests pour path* methods

# Problems
- RoundRobin problem (GCC)
- Job-shop problem 
- Golfers problem
- Car sequencing problem (GCC or Element)
- Crypto multiplication problem (Product)

# CSPlib
- review CSPlib for additional problem

# Engine
- optimizer: implement choice points pruning
- implement a backtrackable propagator __state__
- allow custom propagators

# Numba
- use prange with parallel=True
- test ufunc
- test vectorize

# Docs
- document environment variables
- document the absence of decision variables
- document solvers
- document all functions