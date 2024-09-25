#######################
Reference documentation
#######################


.. _propagators:

***********
Propagators
***********

NUCS currently provides the following highly-optimized propagators.

==================================== ============================================== ========================
Name                                 Definition                                     Complexity
==================================== ============================================== ========================
:code:`affine_eq_propagator`         :math:`\Sigma_i a_i \times x_i = a_{n-1}`      :math:`n`
:code:`affine_geq_propagator`        :math:`\Sigma_i a_i \times x_i \geq a_{n-1}`   :math:`n`
:code:`affine_leq_propagator`        :math:`\Sigma_i a_i \times x_i \leq a_{n-1}`   :math:`n`
:code:`alldifferent_propagator`      :math:`\forall i, j, x_i \neq x_j`             :math:`n \times log(n)`
:code:`count_eq_propagator`          :math:`\Sigma_i (x_i = a) = x_{n-1}`           :math:`n`
:code:`element_lic_propagator`       :math:`l_i = c`                                :math:`n`
:code:`element_liv_propagator`       :math:`l_i = x`                                :math:`n`
:code:`exactly_eq_propagator`        :math:`\Sigma_i (x_i = a_0) = a_1`             :math:`n`
:code:`lexicographic_leq_propagator` :math:`x \leq_{lex} y`                         :math:`n`
:code:`max_eq_propagator`            :math:`\max_i x_i = x_{n-1}`                   :math:`n`
:code:`max_leq_propagator`           :math:`\max_i x_i \leq x_{n-1}`                :math:`n`
:code:`min_eq_propagator`            :math:`\min_i x_i = x_{n-1}`                   :math:`n`
:code:`min_geq_propagator`           :math:`\min_i x_i \geq x_{n-1}`                :math:`n`
:code:`relation_propagator`          Relation defined in extension                  :math:`n`
==================================== ============================================== ========================

.. _heuristics:

**********
Heuristics
**********

NUCS provides heuristics for selecting a variable (precisely selecting a shared domain)
and for selecting a value (more generally, reducing the shared domain):

Heuristics for selecting a shared domain
########################################

NUCS provides the following functions for selecting a shared domain.

============================================ ============================================================
Function                                     Description
============================================ ============================================================
:code:`first_not_instantiated_var_heuristic` selects the first non-instantiated shared domain
:code:`last_not_instantiated_var_heuristic`  selects the last non-instantiated shared domain
:code:`smallest_domain_var_heuristic`        selects the smallest shared domain which is not instantiated
:code:`greatest_domain_var_heuristic`        selects the greatest shared domain which is not instantiated
============================================ ============================================================

Heuristics for reducing the chosen shared domain
################################################

NUCS provides the following functions for reducing a shared domain.

============================================ ============================================================
Function                                     Description
============================================ ============================================================
:code:`min_value_dom_heuristic`              selects the minimal value of the domain
:code:`max_value_dom_heuristic`              selects the maximal value of the domain
:code:`split_low_dom_heuristic`              selects the first half of the domain
============================================ ============================================================

.. _examples:

********
Examples
********

NUCS comes with the following examples.
Some of these examples have a command line interface and can be run directly:

.. code-block:: bash

   NUMBA_CACHE_DIR=.numba/cache PYTHON_PATH=. python -m nucs.examples.<problem> <options>

================================ ======== ======================== =====================================================
Problem                          CSPLib # CLI                      Options
================================ ======== ======================== =====================================================
:code:`alpha`                             Yes
:code:`bibd`                     028      Yes (for one instance)
:code:`donald`                            Yes
:code:`golomb`                   006      Yes                      :code:`-n` size
:code:`knapsack`                 133      Yes (for one instance)
:code:`magic_sequence`           019
:code:`magic_square`             019
:code:`quasigroup`               003      Yes (for one subproblem) :code:`-n` size
:code:`queens`                   054      Yes                      :code:`-n` size
:code:`schur_lemma`              015
:code:`sudoku`
================================ ======== ======================== =====================================================

