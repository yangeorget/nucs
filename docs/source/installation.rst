############
Installation
############

.. _installation:

************************
Install the NUCS package
************************

Let's install the NUCS package with pip:

.. code-block:: bash

   $ pip install nucs

Now we can write the following :code:`queens.py` program:

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

*****************************
Install NUCS from the sources
*****************************

Let's install NUCS from the sources by cloning the NUCS Github repository:

.. code-block:: bash

   git clone https://github.com/yangeorget/nucs.git
   pip install -r requirements.txt

Some of the examples come with a command line interface and can be run directly.

Let's find all solutions to the `12-queens problem <https://www.csplib.org/Problems/prob054>`_:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_queens.py -n 12

Let's find the optimal solution to the `Golomb ruler problem <https://www.csplib.org/Problems/prob006>`_ with 10 marks:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. python tests/examples/test_golomb.py -n 10






