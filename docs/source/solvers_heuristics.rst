######################
Solvers and heuristics
######################

*******
Solvers
*******

NuCS comes with some pre-defined :ref:`solvers <solvers>`.

Backtracking-based solver
#########################
NuCS provides :mod:`nucs.solvers.backtrack_solver` which is the main solver.

Multiprocessing-based solver
############################
NuCS also provides :mod:`nucs.solvers.multiprocessing_solver` which relies on the Python :code:`multiprocessing` package.

This solver is used by the launcher of the :mod:`nucs.examples.queens.queens_problem`.

.. code-block:: python
   :linenos:

   problem = QueensProblem(args.n)
   problems = problem.split(args.processors, 0)  # creates n-subproblems by splitting the domain of the first variable
   solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
   solver.solve_all()


**********
Heuristics
**********

NuCS comes with some pre-defined :ref:`heuristics <heuristics>` and makes it possible to design custom heuristics.

Custom heuristics
#################
NuCS makes it possible to define and use custom heuristics.

A heuristic needs to be registered before it is used.
The following code registers the :code:`SPLIT_LOW` heuristic.

.. code-block:: python
   :linenos:

   DOM_HEURISTIC_SPLIT_LOW = register_dom_heuristic(split_low_dom_heuristic)


