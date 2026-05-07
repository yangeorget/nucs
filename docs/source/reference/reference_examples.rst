.. _examples:

********
Examples
********

NUCS comes with several examples.

Most of these examples can be run from the command line and support the following options:

* :code:`--consistency`: set the consistency algorithm (0 is for BC, 1 for BC+shaving), defaults to BC
* :code:`--cp-max-height`: set the maximal height of the choice points stack, defaults to 512
* :code:`--dataset`: the dataset to use
* :code:`--display-solutions`: display the solution(s), defaults to true
* :code:`--display-stats`: display the statistics, defaults to true
* :code:`--find-all`: find all solutions, defaults to false
* :code:`--help`: show the help
* :code:`--log-level`: set the log level, can take the values :code:`DEBUG`, :code:`INFO`, :code:`WARNING`, :code:`ERROR`, :code:`CRITICAL`, defaults to :code:`INFO`
* :code:`--n`: define the size of the problem
* :code:`--optimization-mode`: set the optimizer mode (:code:`RESET` or :code:`PRUNE`), defaults to :code:`RESET`
* :code:`--processors`: define the number of processors to use
* :code:`--symmetry-breaking/--no-symmetry-breaking`: leverage symmetries in the problem, defaults to true


.. autoclass:: nucs.examples.all_interval_series.all_interval_series_problem.AllIntervalSeriesProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.all_interval_series

This problem leverages the propagators:

* :mod:`nucs.propagators.abs_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.leq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. autoclass:: nucs.examples.alphanumeric.alphanumeric_problem.AlphanumericProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alphanumeric

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. autoclass:: nucs.examples.bibd.bibd_problem.BIBDProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.and_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. autoclass:: nucs.examples.cryptarithmetic.cryptarithmetic_problem.CryptarithmeticProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.cryptarithmetic

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. autoclass:: nucs.examples.employee_scheduling.employee_scheduling_problem.EmployeeSchedulingProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.employee_scheduling

This problem leverages the propagators:

* :mod:`nucs.propagators.count_eq_c_propagator`,
* :mod:`nucs.propagators.count_eq_propagator`,
* :mod:`nucs.propagators.count_leq_c_propagator`.


.. autoclass:: nucs.examples.golomb.golomb_problem.GolombProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.leq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. autoclass:: nucs.examples.knapsack.knapsack_problem.KnapsackProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.knapsack

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`.


.. autoclass:: nucs.examples.langford.langford_problem.LangfordProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.langford

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. autoclass:: nucs.examples.magic_sequence.magic_sequence_problem.MagicSequenceProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.count_eq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. autoclass:: nucs.examples.magic_square.magic_square_problem.MagicSquareProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_square

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.leq_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.


.. autoclass:: nucs.examples.quasigroup.quasigroup_problem.QuasigroupProblem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup

This quasigroup problem leverages the problem :mod:`nucs.problems.latin_square_problem` and the propagators:

* :mod:`nucs.propagators.element_liv_alldifferent_propagator`.


.. autoclass:: nucs.examples.queens.queens_problem.QueensProblem

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.


.. autoclass:: nucs.examples.schur_lemma.schur_lemma_problem.SchurLemmaProblem

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. autoclass:: nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem.SportsTournamentSchedulingProblem

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.sports_tournament_scheduling

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`,
* :mod:`nucs.propagators.exactly_eq_propagator`,
* :mod:`nucs.propagators.gcc_propagator`,
* :mod:`nucs.propagators.relation_propagator`.


.. autoclass:: nucs.examples.sudoku.sudoku_problem.SudokuProblem

This problem leverages the :mod:`nucs.problems.latin_square_problem` and the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.


.. autoclass:: nucs.examples.tsp.tsp_problem.TSPProblem

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.tsp

This problem leverages the :mod:`nucs.problems.circuit_problem` and the propagators:

* :mod:`nucs.propagators.element_iv_propagator`,
* :mod:`nucs.propagators.sum_eq_propagator`.
