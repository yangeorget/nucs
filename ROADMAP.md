# Constraints
- Cycle
- Product

# Problems
- Job-shop problem 
- Golfers problem
- Car sequencing problem (GCC or Element)
- Crypto multiplication problem (Product)

# CSPlib
- review CSPlib for additional problem

# Engine
- implement a backtrackable propagator __state__ and optimize GCC and LexLeq
- optimizer: implement choice points pruning
- allow custom propagators
- assert that parameters make sense (cf GCC)
- compare perfs with KCS
- study multiprocessing

# Docs
- document environment variables
- document the absence of decision variables
- document solvers