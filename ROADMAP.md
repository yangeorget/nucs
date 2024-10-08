# Constraints
- GCC:
  - write a smaller test that loops
  - compare with KCS !!!
  
- Cycle
- Product

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
- assert that parameters make sense (cf GCC)

# Numba
- use prange with parallel=True
- test ufunc
- test vectorize

# Docs
- document environment variables
- document the absence of decision variables
- document solvers