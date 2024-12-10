#######
Solvers
#######

NuCS comes with some pre-defined :ref:`solvers <solvers>`.

****************
Solver arguments
****************

A solver accepts the following parameters:

* the problem to be solved
* the logging level


*************************
Backtracking-based solver
*************************

NuCS provides :mod:`nucs.solvers.backtrack_solver` which is the main solver.


Backtracking solver arguments
#############################

A backtrracking solver accepts the additional following parameters:

* the consistency algorithm to use (bound consistency is used by default)
* the decision variables/shared domains (all are used by default)
* an heuristic to choose a variable/shared domain (the first non instantiated is chosen by default)
* some parameters for this heuristic (none by default)
* an heuristic to select a value (the first value is chosen by default)
* some parameters for this heuristic (none by default)
* the maximal height for the choice points stack (128 by default)


****************************
Multiprocessing-based solver
****************************
NuCS also provides :mod:`nucs.solvers.multiprocessing_solver` which relies on the Python :code:`multiprocessing` package.

This solver is used by the launcher of the :mod:`nucs.examples.queens.queens_problem`.

.. code-block:: python
   :linenos:

   problem = QueensProblem(args.n)
   problems = problem.split(args.processors, 0)  # creates n-subproblems by splitting the domain of the first variable
   solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
   solver.solve_all()

