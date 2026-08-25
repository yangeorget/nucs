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
* a list of searches, each with its own decision variables and heuristics (a single search by default)
* the maximal height for the choice points stack (8192 by default)


*******************
Searching solutions
*******************

A solver exposes two searches, satisfaction and optimization, each in three forms:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Satisfaction
     - Optimization
     - Returns
   * - :code:`solve`
     - :code:`optimize`
     - an iterator over the solutions
   * - :code:`solve_all`
     - :code:`optimize_all`
     - nothing, calling a function on each solution found
   * - :code:`find_all`
     - :code:`find_best`
     - the list of all solutions / the best solution

The optimization methods take the variable to optimize and a bound: :code:`MIN` to minimize it,
:code:`MAX` to maximize it.

.. code-block:: python
   :linenos:

   from nucs.constants import MIN

   solution = solver.find_best(problem.total_cost, MIN)


*******
Timeout
*******

Every search method accepts an optional :code:`timeout`, a wall-clock budget in seconds.
When it is exhausted the search stops and the solver's :code:`timed_out` attribute is set,
which is what distinguishes an exhausted search from a truncated one:

.. code-block:: python
   :linenos:

   solution = solver.find_best(problem.total_cost, MIN, timeout=60)
   if solver.timed_out:
       print("best solution found within the budget, not proven optimal")

.. note::
   The budget can only be checked where the search returns to Python, that is between two solutions,
   because a single descent runs in compiled code that nothing in the process can interrupt.
   It therefore bounds the time spent *between* solutions, not the time spent inside one descent.

