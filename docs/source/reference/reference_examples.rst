.. _examples:

********
Examples
********

NUCS comes with several examples.

Most of these examples can be run from the command line and support the following options:

* :code:`--consistency`: set the consistency algorithm (0 is for BC, 1 for BC+shaving)
* :code:`--cp-max-height`: set the maximal height of the choice points stack (default is 512)
* :code:`--display-solutions`: display the solution(s)
* :code:`--display-stats`: display the statistics
* :code:`--find-all`: find all solutions
* :code:`--help`: show the help
* :code:`--log-level`: set the log level, can take the values :code:`DEBUG`, :code:`INFO`, :code:`WARNING`, :code:`ERROR`, :code:`CRITICAL`
* :code:`--n`: define the size of the problem
* :code:`--optimization-mode`: set the optimizer mode (:code:`RESET` or :code:`PRUNE`)
* :code:`--processors`: define the number of processors to use
* :code:`--symmetry-breaking/--no-symmetry-breaking`: leverage symmetries in the problem


.. py:module:: nucs.examples.all_interval_series.all_interval_series_problem
.. py:class:: nucs.examples.all_interval_series.all_interval_series_problem.AllIntervalSeriesProblem

This problem is problem `007 <https://www.csplib.org/Problems/prob007>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.all_interval_series

This problem leverages the propagators:

* :mod:`nucs.propagators.abs_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.leq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. py:module:: nucs.examples.alpha.alpha_problem
.. py:class:: nucs.examples.alpha.alpha_problem.AlphaProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alpha

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.bibd.bibd_problem
.. py:class:: nucs.examples.bibd.bibd_problem.BIBDProblem

This problem is problem `028 <https://www.csplib.org/Problems/prob028>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.and_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. py:module:: nucs.examples.donald.donald_problem
.. py:class:: nucs.examples.donald.donald_problem.DonaldProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.donald

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


. py:module:: nucs.examples.employee_scheduling.employee_scheduling_problem
.. py:class:: nucs.examples.employee_scheduling.employee_scheduling.EmployeeSchedulingProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.employee_scheduling

This problem leverages the propagators:

* :mod:`nucs.propagators.count_eq_c_propagator`,
* :mod:`nucs.propagators.count_eq_propagator`,
* :mod:`nucs.propagators.count_leq_c_propagator`.


.. py:module:: nucs.examples.golomb.golomb_problem
.. py:class:: nucs.examples.golomb.golomb_problem.GolombProblem

This problem is problem `006 <https://www.csplib.org/Problems/prob006>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.leq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. py:module:: nucs.examples.knapsack.knapsack_problem
.. py:class:: nucs.examples.knapsack.knapsack_problem.KnapsackProblem

This problem is problem `133 <https://www.csplib.org/Problems/prob133>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.knapsack

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`.


.. py:module:: nucs.examples.magic_sequence.magic_sequence_problem
.. py:class:: nucs.examples.magic_sequence.magic_sequence_problem.MagicSequenceProblem

This problem is problem `019 <https://www.csplib.org/Problems/prob019>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.count_eq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. py:module:: nucs.examples.magic_square.magic_square_problem
.. py:class:: nucs.examples.magic_square.magic_square_problem.MagicSquareProblem

This problem is problem `019 <https://www.csplib.org/Problems/prob019>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_square

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.leq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. py:module:: nucs.examples.quasigroup.quasigroup_problem
.. py:class:: nucs.examples.quasigroup.quasigroup_problem.QuasigroupProblem

This problem is problem `003 <https://www.csplib.org/Problems/prob003>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup

This quasigroup problem leverages the problem :mod:`nucs.problems.latin_square_problem` and the propagators:

* :mod:`nucs.propagators.element_liv_alldifferent_propagator`.


.. py:module:: nucs.examples.queens.queens_problem
.. py:class:: nucs.examples.queens.queens_problem.QueensProblem

This problem is problem `054 <https://www.csplib.org/Problems/prob054>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.schur_lemma.schur_lemma_problem
.. py:class:: nucs.examples.schur_lemma.schur_lemma_problem.SchurLemmaProblem

This problem is problem `015 <https://www.csplib.org/Problems/prob015>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. py:module:: nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem
.. py:class:: nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem.SportSchedulingTournamentProblem

This problem is problem `026 <https://www.csplib.org/Problems/prob026>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.sports_tournament_scheduling

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.exactly_eq_propagator`,
* :mod:`nucs.propagators.gcc_propagator`,
* :mod:`nucs.propagators.relation_propagator`.


.. py:module:: nucs.examples.sudoku.sudoku_problem
.. py:class:: nucs.examples.sudoku.sudoku_problem.SudokuProblem

This problem leverages the :mod:`nucs.problems.latin_square_problem` and the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.tsp.tsp_problem
.. py:class:: nucs.examples.tsp.tsp_problem.TSPProblem

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.tsp

This problem leverages the :mod:`nucs.problems.circuit_problem` and the propagators:

* :mod:`nucs.propagators.element_iv_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.
