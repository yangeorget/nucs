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
- optimizer: implement choice points pruning
- implement a backtrackable propagator __state__
- implement decision variables
- disable offsets when not useful
- entailment date
- allow custom propagators

# Numba
- use prange with parallel=True
- test ufunc
- test vectorize

# Docs
- document environment variables
- fix doc after adding new constraints