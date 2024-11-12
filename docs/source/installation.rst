############
Installation
############

************************
Install the NuCS package
************************

Let's install the NuCS package with pip:

.. code-block:: bash

   pip install nucs

*****************
Run some examples
*****************

NuCS comes with some models and :ref:`heuristics <heuristics>` for some well-known :ref:`examples <examples>`.
Some of these examples have a command line interface and can be run directly.

Solve the 12-queens problem
###########################
Let's find all solutions to the `12-queens problem <https://www.csplib.org/Problems/prob054>`_:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens -n 12 --log_level=INFO


Produces the following output:

.. code-block:: bash

   2024-11-12 17:24:49,061 - INFO - nucs.solvers.solver - Problem has 3 propagators
   2024-11-12 17:24:49,061 - INFO - nucs.solvers.solver - Problem has 12 variables
   2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses variable heuristic 0
   2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses domain heuristic 0
   2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses consistency algorithm 0
   2024-11-12 17:24:49,061 - INFO - nucs.solvers.backtrack_solver - Choice points stack has a maximal height of 128
   2024-11-12 17:24:49,200 - INFO - nucs.solvers.multiprocessing_solver - MultiprocessingSolver has 1 processors
   {
      'OPTIMIZER_SOLUTION_NB': 0,
      'PROBLEM_FILTER_NB': 262011,
      'PROBLEM_SHAVING_NB': 0,
      'PROBLEM_SHAVING_CHANGE_NB': 0,
      'PROBLEM_SHAVING_NO_CHANGE_NB': 0,
      'PROPAGATOR_ENTAILMENT_NB': 0,
      'PROPAGATOR_FILTER_NB': 2269980,
      'PROPAGATOR_FILTER_NO_CHANGE_NB': 990450,
      'PROPAGATOR_INCONSISTENCY_NB': 116806,
      'SOLVER_BACKTRACK_NB': 131005,
      'SOLVER_CHOICE_NB': 131005,
      'SOLVER_CHOICE_DEPTH': 10,
      'SOLVER_SOLUTION_NB': 14200
   }

Solve the Golomb ruler problem
##############################
Let's find the optimal solution to the `Golomb ruler problem <https://www.csplib.org/Problems/prob006>`_ with 10 marks:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache -m nucs.examples.golomb -n 10 --symmetry_breaking --log_level=INFO


Produces the following output:

.. code-block:: bash

   2024-11-12 17:27:45,110 - INFO - nucs.solvers.solver - Problem has 82 propagators
   2024-11-12 17:27:45,110 - INFO - nucs.solvers.solver - Problem has 45 variables
   2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses variable heuristic 0
   2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses domain heuristic 0
   2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - BacktrackSolver uses consistency algorithm 2
   2024-11-12 17:27:45,110 - INFO - nucs.solvers.backtrack_solver - Choice points stack has a maximal height of 128
   2024-11-12 17:27:45,172 - INFO - nucs.solvers.backtrack_solver - Minimizing variable 8
   2024-11-12 17:27:45,644 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 80
   2024-11-12 17:27:45,677 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 75
   2024-11-12 17:27:45,677 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 73
   2024-11-12 17:27:45,678 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 72
   2024-11-12 17:27:45,679 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 70
   2024-11-12 17:27:45,682 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 68
   2024-11-12 17:27:45,687 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 66
   2024-11-12 17:27:45,693 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 62
   2024-11-12 17:27:45,717 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 60
   2024-11-12 17:27:45,977 - INFO - nucs.solvers.backtrack_solver - Found a (new) solution: 55
   {
       'OPTIMIZER_SOLUTION_NB': 10,
       'PROBLEM_FILTER_NB': 22652,
       'PROBLEM_SHAVING_NB': 0,
       'PROBLEM_SHAVING_CHANGE_NB': 0,
       'PROBLEM_SHAVING_NO_CHANGE_NB': 0,
       'PROPAGATOR_ENTAILMENT_NB': 107911,
       'PROPAGATOR_FILTER_NB': 2813035,
       'PROPAGATOR_FILTER_NO_CHANGE_NB': 1745836,
       'PROPAGATOR_INCONSISTENCY_NB': 11289,
       'SOLVER_BACKTRACK_NB': 11288,
       'SOLVER_CHOICE_NB': 11353,
       'SOLVER_CHOICE_DEPTH': 9,
       'SOLVER_SOLUTION_NB': 10
   }
   [ 1  6 10 23 26 34 41 53 55]

**********************
Write your first model
**********************

Model the n-queens problem
###########################

Let's write the following :code:`queens.py` program:

.. code-block:: python
   :linenos:

   from nucs.problems.problem import Problem
   from nucs.solvers.backtrack_solver import BacktrackSolver
   from nucs.propagators.propagators import ALG_ALLDIFFERENT

   n = 8  # the number of queens
   problem = Problem(
       [(0, n - 1)] * n,  # these n domains are shared between the 3n variables with different offsets
       list(range(n)) * 3,  # for each variable, its shared domain
       [0] * n + list(range(n)) + list(range(0, -n, -1))  # for each variable, its offset
   )
   problem.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
   problem.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
   problem.add_propagator((list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []))
   print(BacktrackSolver(problem).solve_one()[:n])

Let's run this model with the following command:

.. code-block:: bash

   $ NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python queens.py

The first solution found is:

.. code-block:: bash

   [0, 4, 7, 5, 2, 6, 1, 3]

.. note::
   Note that the second run will always be **much faster**
   since the Python code will already have been compiled and cached by Numba.




