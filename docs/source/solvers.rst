#######
Solvers
#######

NuCS comes with some pre-defined :ref:`solvers <solvers>`.

****************
Solver arguments
****************

A solver accepts the following parameters:

* the problem to be solved
* the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)


*************************
Backtracking-based solver
*************************

NuCS provides :mod:`nucs.solvers.backtrack_solver` which is the main solver.


Backtracking solver arguments
#############################

A backtracking solver accepts the additional following parameters:

* the consistency algorithm to use (bound consistency is used by default)
* the decision variables(all are used by default)
* an heuristic to choose a variable (the first non instantiated is chosen by default)
* some parameters for this heuristic (none by default)
* an heuristic to select a value (the first value is chosen by default)
* some parameters for this heuristic (none by default)
* the maximal height for the choice points stack (256 by default)

