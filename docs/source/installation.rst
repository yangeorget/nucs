############
Installation
############

************************
Install the NuCS package
************************

Let's install the NuCS package with pip:

.. code-block:: bash

   pip install nucs


NuCS requires a recent version of Python: 3.12, 3.13 and 3.14 are supported.


*****************
Run some examples
*****************

NuCS comes with some models and :ref:`heuristics <heuristics>` for some well-known :ref:`examples <examples>`.
Some of these examples have a command line interface and can be run directly.


Solve the 12-queens problem
###########################
Let's find all solutions to the `12-queens problem <https://www.csplib.org/Problems/prob054>`_.

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 12 --find-all --no-display-solutions

Produces the following output:

.. code-block:: bash

   [ 2026-09-30 21:51:36,753 | MainProcess | INFO ] nucs.solvers.solver.__init__ - Initializing Solver
   [ 2026-09-30 21:51:36,799 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 3 propagators
   [ 2026-09-30 21:51:36,799 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 12 variables
   [ 2026-09-30 21:51:36,799 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses decision domains [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], variable heuristics [2] and domain heuristics [3]
   [ 2026-09-30 21:51:36,799 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses consistency algorithm 0
   [ 2026-09-30 21:51:36,800 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - The stack of choice points starts at 8192 rows and grows when it runs out
   [ 2026-09-30 21:51:36,800 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - The trail starts at 65536 entries and grows when it runs out
   [ 2026-09-30 21:51:36,809 | MainProcess | INFO ] nucs.solvers.solver.find_all - Returning all solutions
   [ 2026-09-30 21:51:36,809 | MainProcess | INFO ] nucs.solvers.solver.solve_all - Iterating over the solutions
   [ 2026-09-30 21:51:36,809 | MainProcess | INFO ] nucs.solvers.backtrack_solver.solve - Solving and iterating over the solutions
   {
       'ALG_BC_NB': 262011,
       'PROPAGATOR_ENTAILMENT_NB': 39047,
       'PROPAGATOR_FILTER_NB': 1781373,
       'PROPAGATOR_FILTER_NO_CHANGE_NB': 563741,
       'PROPAGATOR_INCONSISTENCY_NB': 116806,
       'SOLVER_BACKTRACK_NB': 131005,
       'SOLVER_CHOICE_NB': 131005,
       'SOLVER_CHOICE_DEPTH': 10,
       'SOLVER_RESTART_NB': 0,
       'SOLUTION_NB': 14200,
       'SOLVER_ELAPSED_TIME_MS': 964,
       'PROPAGATOR_FILTER_NB_ALLDIFFERENT': 1781373,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_ALLDIFFERENT': 563741
   }


Solve the Golomb ruler problem
##############################
Let's find the optimal solution to the `Golomb ruler problem <https://www.csplib.org/Problems/prob006>`_ with 10 marks.

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb -n 10

Produces the following output:

.. code-block:: bash

   [ 2026-09-30 21:51:38,022 | MainProcess | INFO ] nucs.solvers.solver.__init__ - Initializing Solver
   [ 2026-09-30 21:51:38,028 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 82 propagators
   [ 2026-09-30 21:51:38,028 | MainProcess | INFO ] nucs.problems.problem.init - Problem has 45 variables
   [ 2026-09-30 21:51:38,028 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses decision domains [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]], variable heuristics [2] and domain heuristics [3]
   [ 2026-09-30 21:51:38,028 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - BacktrackSolver uses consistency algorithm 1
   [ 2026-09-30 21:51:38,029 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - The stack of choice points starts at 8192 rows and grows when it runs out
   [ 2026-09-30 21:51:38,029 | MainProcess | INFO ] nucs.solvers.backtrack_solver.__init__ - The trail starts at 65536 entries and grows when it runs out
   [ 2026-09-30 21:51:38,039 | MainProcess | INFO ] nucs.solvers.solver.find_best - Returning the optimal solution
   [ 2026-09-30 21:51:38,039 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Optimizing and iterating over the solutions
   [ 2026-09-30 21:51:38,043 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 80
   [ 2026-09-30 21:51:38,044 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 75
   [ 2026-09-30 21:51:38,044 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 73
   [ 2026-09-30 21:51:38,045 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 72
   [ 2026-09-30 21:51:38,045 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 70
   [ 2026-09-30 21:51:38,045 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 68
   [ 2026-09-30 21:51:38,046 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 66
   [ 2026-09-30 21:51:38,047 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 62
   [ 2026-09-30 21:51:38,051 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 60
   [ 2026-09-30 21:51:38,093 | MainProcess | INFO ] nucs.solvers.backtrack_solver.optimize - Found a local optimum: 55
   {
       'ALG_BC_NB': 22230,
       'PROPAGATOR_ENTAILMENT_NB': 203454,
       'PROPAGATOR_FILTER_NB': 1400872,
       'PROPAGATOR_FILTER_NO_CHANGE_NB': 586754,
       'PROPAGATOR_INCONSISTENCY_NB': 11078,
       'SOLVER_BACKTRACK_NB': 11077,
       'SOLVER_CHOICE_NB': 11142,
       'SOLVER_CHOICE_DEPTH': 9,
       'SOLVER_RESTART_NB': 0,
       'SOLUTION_NB': 10,
       'SOLVER_ELAPSED_TIME_MS': 127,
       'PROPAGATOR_FILTER_NB_ALLDIFFERENT': 44683,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_ALLDIFFERENT': 6484,
       'PROPAGATOR_FILTER_NB_LEQ_C': 360228,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_LEQ_C': 347920,
       'PROPAGATOR_FILTER_NB_SUM_EQ': 995961,
       'PROPAGATOR_FILTER_NO_CHANGE_NB_SUM_EQ': 232350
   }
   [1, 6, 10, 23, 26, 34, 41, 53, 55, 5]


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

   [0 4 7 5 2 6 1 3]

.. note::
   Note that the second run will always be **much faster**
   since the Python code will already have been compiled and cached by Numba.




