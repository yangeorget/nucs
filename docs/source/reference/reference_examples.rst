.. _examples:

********
Examples
********

NUCS comes with the following examples.


.. py:module:: nucs.examples.alpha.alpha_problem
.. py:class:: nucs.examples.alpha.alpha_problem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alpha

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.bibd.bibd_problem
.. py:class:: nucs.examples.bibd.bibd_problem

This problem is problem `028 <https://www.csplib.org/Problems/prob028>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.bibd -v 8 -b 14 -r 7 -k 4 -l 3

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.and_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. py:module:: nucs.examples.donald.donald_problem
.. py:class:: nucs.examples.donald.donald_problem

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.donald

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.golomb.golomb_problem
.. py:class:: nucs.examples.golomb.golomb_problem

This problem is problem `006 <https://www.csplib.org/Problems/prob006>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.golomb

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.knapsack.knapsack_problem
.. py:class:: nucs.examples.knapsack.knapsack_problem

This problem is problem `133 <https://www.csplib.org/Problems/prob133>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.knapsack

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`.


.. py:module:: nucs.examples.magic_sequence.magic_sequence_problem
.. py:class:: nucs.examples.magic_sequence.magic_sequence_problem

This problem is problem `019 <https://www.csplib.org/Problems/prob019>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_sequence

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.count_eq_propagator`.


.. py:module:: nucs.examples.magic_square.magic_square_problem
.. py:class:: nucs.examples.magic_square.magic_square_problem

This problem is problem `019 <https://www.csplib.org/Problems/prob019>`_ on CSPLib.

This problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.magic_square

This problem leverages the propagators:

* :mod:`nucs.propagators.affine_eq_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.quasigroup.quasigroup_problem
.. py:class:: nucs.examples.quasigroup.quasigroup_problem

This problem is problem `003 <https://www.csplib.org/Problems/prob003>`_ on CSPLib.

The problem QG5, a sub-instance of the quasigroup problem, can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.quasigroup

This problem leverages the propagators:

* :mod:`nucs.propagators.element_liv_propagator`,
* :mod:`nucs.propagators.element_lic_propagator`,
* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.queens.queens_problem
.. py:class:: nucs.examples.queens.queens_problem

This problem is problem `054 <https://www.csplib.org/Problems/prob054>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.queens

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.


.. py:module:: nucs.examples.schur_lemma.schur_lemma_problem
.. py:class:: nucs.examples.schur_lemma.schur_lemma_problem

This problem is problem `015 <https://www.csplib.org/Problems/prob015>`_ on CSPLib.

The problem can be run with the command:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.schur_lemma

This problem leverages the propagators:

* :mod:`nucs.propagators.exactly_true_propagator`,
* :mod:`nucs.propagators.affine_leq_propagator`,
* :mod:`nucs.propagators.lexicographic_leq_propagator`.


.. py:module:: nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem
.. py:class:: nucs.examples.sports_tournament_scheduling.sports_tournament_scheduling_problem

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
.. py:class:: nucs.examples.sudoku.sudoku_problem

This problem leverages the propagators:

* :mod:`nucs.propagators.alldifferent_propagator`.