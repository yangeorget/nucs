#######################
Reference documentation
#######################


.. _propagators:

***********
Propagators
***********

NUCS currently provides the following highly-optimized propagators.


.. py:function:: nucs.propagators.affine_eq_propagator(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i = a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray


.. py:function:: nucs.propagators.affine_geq_propagator(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i \geq a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray


.. py:function:: nucs.propagators.affine_leq_propagator(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i \leq a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray


.. py:function:: nucs.propagators.alldifferent_propagator(domains, parameters)

   This propagator implements the relation :math:`\forall i \neq j, x_i \neq x_j`.

   It is adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".

   It has the time complexity: :math:`O(n \times log(n))` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray

.. py:function:: nucs.propagators.count_eq_propagator(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (x_i = a) = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is the first parameter
   :type parameters: NDArray


.. py:function:: nucs.propagators.element_lic_propagator(domains, parameters)

   This propagator implements the relation :math:`l_i = c`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-1` domains,
          :math:`i` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`c` is the first parameter
   :type parameters: NDArray


.. py:function:: nucs.propagators.element_liv_propagator(domains, parameters)

   This propagator implements the relation :math:`l_i = v`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-2` domains,
          :math:`i` is the :math:`n-1` th domain,
          :math:`v` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:function:: nucs.propagators.exactly_eq_propagator(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (x_i = a) = c`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          :math:`a` is the first parameter,
          :math:`c` is the second parameter
   :type parameters: NDArray


.. py:function:: nucs.propagators.lexicographic_leq_propagator(domains, parameters)

   This propagator implements the relation :math:`x <_{leq} y`.

   See https://www.diva-portal.org/smash/record.jsf?pid=diva2:1041533.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is the list of the first :math:`n` domains,
          :math:`y` is the list of the last :math:`n` domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:function:: nucs.propagators.max_eq_propagator(domains, parameters)

   This propagator implements the relation :math:`\max_i x_i = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:function:: nucs.propagators.max_leq_propagator(domains, parameters)

   This propagator implements the relation :math:`\max_i x_i \leq x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:function:: nucs.propagators.min_eq_propagator(domains, parameters)

   This propagator implements the relation :math:`\min_i x_i = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:function:: nucs.propagators.min_geq_propagator(domains, parameters)

   This propagator implements the relation :math:`\min_i x_i \geq x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray


.. py:function:: nucs.propagators.relation_propagator(domains, parameters)

   This propagator implements a relation over :math:`O(n)` variables defined by its allowed tuples.

   It has the time complexity: :math:`O(p)` where :math:`p` is the number of parameters.

   :param domains: the domains of the variables
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          the allowed tuples correspond to:
          :math:`(p_0, ..., p_{n-1}), (p_n, ..., p_{2n-1}), ...` where :math:`p` is an alias for parameters

   :type parameters: NDArray


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

