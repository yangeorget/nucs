.. _propagators:

***********
Propagators
***********

NuCS currently provides the following highly-optimized propagators.


.. py:module:: nucs.propagators.affine_eq_propagator
.. py:function:: nucs.propagators.affine_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i = a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.affine_geq_propagator
.. py:function:: nucs.propagators.affine_geq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i \geq a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.affine_leq_propagator
.. py:function:: nucs.propagators.affine_leq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_{i \in [0, n-1[} a_i \times x_i \leq a_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is an alias for parameters
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.and_propagator
.. py:function:: nucs.propagators.and_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\&_{i \in [0, n-1[} b_i = b_{n-1}`
   where for each :math:`i`, :math:`b_i` is a boolean variable.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`b` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.alldifferent_propagator
.. py:function:: nucs.propagators.alldifferent_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\forall i \neq j, x_i \neq x_j`.

   It is adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".

   It has the time complexity: :math:`O(n \times log(n))` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.count_eq_propagator
.. py:function:: nucs.propagators.count_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (x_i = a) = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`a` is the first parameter
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.element_iv_propagator
.. py:function:: nucs.propagators.element_iv_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`l_i = v` where :math:`l` is a list of constants,
   :math:`i` and :math:`v` two variables.

   It has the time complexity: :math:`O(1)`.

   :param domains: the domains of the variables,
          :math:`i` is the first domain,
          :math:`v` is the second domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`l` is an alias for parameters
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.element_lic_alldifferent_propagator
.. py:function:: nucs.propagators.element_lic_alldifferent_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`l_i = c` where :math:`l` is a list of variables that are all different,
   :math:`i` a variable and :math:`c` a constant.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-1` domains,
          :math:`i` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`c` is the first parameter
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.element_lic_propagator
.. py:function:: nucs.propagators.element_lic_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`l_i = c` where :math:`l` is a list of variables,
   :math:`i` a variable and :math:`c` a constant.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-1` domains,
          :math:`i` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, :math:`c` is the first parameter
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.element_liv_alldifferent_propagator
.. py:function:: nucs.propagators.element_liv_alldifferent_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`l_i = v` where :math:`l` is a list of variables that are all different,
   :math:`i` and :math:`v` two variables.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-2` domains,
          :math:`i` is the :math:`n-1` th domain,
          :math:`v` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.element_liv_propagator
.. py:function:: nucs.propagators.element_liv_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`l_i = v` where :math:`l` is a list of variables,
   :math:`i` and :math:`v` two variables.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`l` is the list of the first :math:`n-2` domains,
          :math:`i` is the :math:`n-1` th domain,
          :math:`v` is the last domain
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.exactly_eq_propagator
.. py:function:: nucs.propagators.exactly_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (x_i = a) = c`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          :math:`a` is the first parameter,
          :math:`c` is the second parameter
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.exactly_true_propagator
.. py:function:: nucs.propagators.exactly_true_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\Sigma_i (b_i = 1) = c`
   where for each :math:`i`, :math:`b_i` is a boolean variable.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`b` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          :math:`c` is the first parameter
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.gcc_propagator
.. py:function:: nucs.propagators.gcc_propagator.compute_domains(domains, parameters)

   This propagator (Global Cardinality Constraint) enforces that
   :math:`\forall j,  l_j \leq |\{ i  / x_i = v_j \}| \leq v_j`.

   It is adapted from "A fast and simple algorithm for bounds consistency of the alldifferent constraint".

   It has the time complexity: :math:`O(n \times log(n))` where :math:`n` is the number of variables.

   :param domains: the domains of the variables, :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, there are :math:`1 + 2 \times m` parameters:
    the first domain value :math:`v_0`, then the :math:`m` lower bounds, then the :math:`m` upper bounds (capacities)
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.lexicographic_leq_propagator
.. py:function:: nucs.propagators.lexicographic_leq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`x <_{leq} y`.

   See https://www.diva-portal.org/smash/record.jsf?pid=diva2:1041533.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is the list of the first :math:`n` domains,
          :math:`y` is the list of the last :math:`n` domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.max_eq_propagator
.. py:function:: nucs.propagators.max_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\max_i x_i = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.max_leq_propagator
.. py:function:: nucs.propagators.max_leq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\max_i x_i \leq x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.min_eq_propagator
.. py:function:: nucs.propagators.min_eq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\min_i x_i = x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.min_geq_propagator
.. py:function:: nucs.propagators.min_geq_propagator.compute_domains(domains, parameters)

   This propagator implements the relation :math:`\min_i x_i \geq x_{n-1}`.

   It has the time complexity: :math:`O(n)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
          :math:`x` is an alias for domains
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.no_sub_cycle_propagator
.. py:function:: nucs.propagators.no_sub_cycle_propagator.compute_domains(domains, parameters)

   This propagator enforces that a permutation does not contain any sub-cycle.

   It has the time complexity: :math:`O(nˆ2)` where :math:`n` is the number of variables.

   :param domains: the domains of the variables,
   :type domains: NDArray
   :param parameters: the parameters of the propagator, it is unused
   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int


.. py:module:: nucs.propagators.relation_propagator
.. py:function:: nucs.propagators.relation_propagator.compute_domains(domains, parameters)

   This propagator implements a relation over :math:`O(n)` variables defined by its allowed tuples.

   It has the time complexity: :math:`O(p)` where :math:`p` is the number of parameters.

   :param domains: the domains of the variables
   :type domains: NDArray
   :param parameters: the parameters of the propagator,
          the allowed tuples correspond to:
          :math:`(p_0, ..., p_{n-1}), (p_n, ..., p_{2n-1}), ...` where :math:`p` is an alias for parameters

   :type parameters: NDArray
   :return: the status of the propagation (consistency, inconsistency or entailment)
   :rtype: int
