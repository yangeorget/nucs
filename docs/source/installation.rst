############
Installation
############

************************
Install the NuCS package
************************

Let's install the NuCS package with pip:

.. code-block:: bash

   pip install nucs


NuCS requires a recent version of Python: 3.11, 3.12 and 3.13 are supported.


*****************
Run some examples
*****************

NuCS comes with some models and :ref:`heuristics <heuristics>` for some well-known :ref:`examples <examples>`.
Some of these examples have a command line interface and can be run directly.


Solve the 12-queens problem
###########################
Let's find all solutions to the `12-queens problem <https://www.csplib.org/Problems/prob054>`_.

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 12

Produces the following output:

.. code-block:: bash

   [ 2025-01-27 14:55:53,114 | MainProcess | INFO ] nucs.solvers.solver.__init__ - Initializing Solver
   [ 2025-01-27 14:55:53,115 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 3 propagators
   [ 2025-01-27 14:55:53,115 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 12 variables
   [ 2025-01-27 14:55:53,115 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses decision domains [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
   [ 2025-01-27 14:55:53,115 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses variable heuristic 0
   [ 2025-01-27 14:55:53,115 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses domain heuristic 3
   [ 2025-01-27 14:55:53,115 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses consistency algorithm 0
   [ 2025-01-27 14:55:53,115 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - The stacks of the choice points have a maximal height of 256
   [ 2025-01-27 14:55:53,196 | MainProcess | INFO ] nucs.solvers.backtrack_solver.solve - Solving and iterating over the solutions
   {
       'ALG_BC_NB': 262011,
       'ALG_BC_WITH_SHAVING_NB': 0,
       'ALG_SHAVING_NB': 0,
       'ALG_SHAVING_CHANGE_NB': 0,
       'ALG_SHAVING_NO_CHANGE_NB': 0,
       'PROPAGATOR_ENTAILMENT_NB': 0,
       'PROPAGATOR_FILTER_NB': 2269980,
       'PROPAGATOR_FILTER_NO_CHANGE_NB': 990450,
       'PROPAGATOR_INCONSISTENCY_NB': 116806,
       'SOLVER_BACKTRACK_NB': 131005,
       'SOLVER_CHOICE_NB': 131005,
       'SOLVER_CHOICE_DEPTH': 10,
       'SOLUTION_NB': 14200,
       'SEARCH_SPACE_INITIAL_SZ': 0,
       'SEARCH_SPACE_REMAINING_SZ': 0,
       'SEARCH_SPACE_LOG2_SCALE': 0
   }


Solve the Golomb ruler problem
##############################
Let's find the optimal solution to the `Golomb ruler problem <https://www.csplib.org/Problems/prob006>`_ with 10 marks.

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache -m nucs.examples.golomb -n 10

Produces the following output:

.. code-block:: bash

   [ 2025-01-27 14:58:17,101 | MainProcess | INFO ] nucs.solvers.solver.__init__ - Initializing Solver
   [ 2025-01-27 14:58:17,101 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 82 propagators
   [ 2025-01-27 14:58:17,102 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 45 variables
   [ 2025-01-27 14:58:17,102 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses decision domains [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
   [ 2025-01-27 14:58:17,102 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses variable heuristic 0
   [ 2025-01-27 14:58:17,102 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses domain heuristic 3
   [ 2025-01-27 14:58:17,102 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses consistency algorithm 2
   [ 2025-01-27 14:58:17,102 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - The stacks of the choice points have a maximal height of 256
   [ 2025-01-27 14:58:17,109 | MainProcess | INFO ] nucs.solvers.backtrack_solver.minimize - Minimizing (mode PRUNE) variable 8 (domain [  1 405]))
   [ 2025-01-27 14:58:17,545 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 80
   [ 2025-01-27 14:58:17,546 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 75
   [ 2025-01-27 14:58:17,547 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 73
   [ 2025-01-27 14:58:17,547 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 72
   [ 2025-01-27 14:58:17,548 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 70
   [ 2025-01-27 14:58:17,549 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 68
   [ 2025-01-27 14:58:17,551 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 66
   [ 2025-01-27 14:58:17,553 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 62
   [ 2025-01-27 14:58:17,571 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 60
   [ 2025-01-27 14:58:17,748 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 55
   {
       'ALG_BC_NB': 20780,
       'ALG_BC_WITH_SHAVING_NB': 0,
       'ALG_SHAVING_NB': 0,
       'ALG_SHAVING_CHANGE_NB': 0,
       'ALG_SHAVING_NO_CHANGE_NB': 0,
       'PROPAGATOR_ENTAILMENT_NB': 115080,
       'PROPAGATOR_FILTER_NB': 2829457,
       'PROPAGATOR_FILTER_NO_CHANGE_NB': 1794797,
       'PROPAGATOR_INCONSISTENCY_NB': 10377,
       'SOLVER_BACKTRACK_NB': 10376,
       'SOLVER_CHOICE_NB': 10393,
       'SOLVER_CHOICE_DEPTH': 9,
       'SOLUTION_NB': 10,
       'SEARCH_SPACE_INITIAL_SZ': 0,
       'SEARCH_SPACE_REMAINING_SZ': 0,
       'SEARCH_SPACE_LOG2_SCALE': 0
   }
   [ 1  6 10 23 26 34 41 53 55]


**********************
Write your first model
**********************

Model the n-queens problem
###########################

Let's write the following :code:`queens.py` program:

.. literalinclude:: queens.py
   :linenos:


Let's run this model with the following command:

.. code-block:: bash

   $ NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python queens.py

The first solution found is:

.. code-block:: bash

   [0, 4, 7, 5, 2, 6, 1, 3]

.. note::
   Note that the second run will always be **much faster**
   since the Python code will already have been compiled and cached by Numba.




