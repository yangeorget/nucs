.. _consistency_algorithms:

**********************
Consistency algorithms
**********************

NuCS provides the following consistency algorithms.

.. py:module:: nucs.solvers.bound_consistency_algorithm
.. py:function:: nucs.solvers.bound_consistency_algorithm.bound_consistency_algorithm(algorithm_nb, propagator_nb, statistics, algorithms, bounds, propagator_variables, propagator_parameters, triggers, domains_stack, entailed_propagators_stack, domain_update_stack, unbound_variable_nb_stk, stacks_top, triggered_propagators, compute_domains_addrs, decision_variables)

   This is the default consistency algorithm used by the :mod:`nucs.solvers.backtrack_solver`.

   :param algorithm_nb: the total number of algorithms
   :type algorithm_nb: int
   :param propagator_nb: the total number of propagators
   :type propagator_nb: int
   :param statistics: a Numpy array of statistics
   :type statistics: NDArray
   :param algorithms: the algorithms indexed by propagators
   :type algorithms: NDArray
   :param bounds: the bounds indexed by propagators
   :type bounds: NDArray
   :param propagator_variables: the variables by propagators
   :type propagator_variables: NDArray
   :param propagator_parameters: the parameters by propagators
   :type propagator_parameters: NDArray
   :param triggers: a Numpy array of event masks indexed by variables
   :type triggers: NDArray
   :param domains_stk: a stack of domains
   :type domains_stk: NDArray
   :param entailed_propagators_stk: a stack of entailed propagators
   :type entailed_propagators_stk: NDArray
   :param domain_update_stk: a stack of domain updates (domain index and bound)
   :type domain_update_stk: NDArray
   :param unbound_variable_nb_stk: a stack of unbound variable number
   :type unbound_variable_nb_stk: NDArray
   :param stks_top: the height of the stacks as a Numpy array
   :type stks_top: NDArray
   :param triggered_propagators: the Numpy array of triggered propagators
   :type triggered_propagators: NDArray
   :param compute_domains_addrs: the addresses of the compute_domains functions
   :type compute_domains_addrs: NDArray
   :param decision_variables: the decision variables
   :type decision_variables: NDArray
   :return: a status (consistency, inconsistency or entailment) as an integer
   :rtype: int


.. py:module:: nucs.solvers.shaving_consistency_algorithm
.. py:function:: nucs.solvers.shaving_consistency_algorithm.shaving_consistency_algorithm(algorithm_nb, propagator_nb, statistics, algorithms, bounds, propagator_variables, propagator_parameters, triggers, domains_stack, entailed_propagators_stack, domain_update_stack, unbound_variable_nb_stk, stacks_top, triggered_propagators, compute_domains_addrs, decision_variables)

   This algorithm reduces the need of searching by shaving the domains.

   :param algorithm_nb: the total number of algorithms
   :type algorithm_nb: int
   :param propagator_nb: the total number of propagators
   :type propagator_nb: int
   :param statistics: a Numpy array of statistics
   :type statistics: NDArray
   :param algorithms: the algorithms indexed by propagators
   :type algorithms: NDArray
   :param bounds: the bounds indexed by propagators
   :type bounds: NDArray
   :param propagator_variables: the variables by propagators
   :type propagator_variables: NDArray
   :param propagator_parameters: the parameters by propagators
   :type propagator_parameters: NDArray
   :param triggers: a Numpy array of event masks indexed by variables
   :type triggers: NDArray
   :param domains_stk: a stack of domains
   :type domains_stk: NDArray
   :param entailed_propagators_stk: a stack of entailed propagators
   :type entailed_propagators_stk: NDArray
   :param domain_update_stk: a stack of domain updates (domain index and bound)
   :type domain_update_stk: NDArray
   :param unbound_variable_nb_stk: a stack of unbound variable number
   :type unbound_variable_nb_stk: NDArray
   :param stks_top: the height of the stacks as a Numpy array
   :type stks_top: NDArray
   :param triggered_propagators: the Numpy array of triggered propagators
   :type triggered_propagators: NDArray
   :param compute_domains_addrs: the addresses of the compute_domains functions
   :type compute_domains_addrs: NDArray
   :param decision_variables: the decision variables
   :type decision_variables: NDArray
   :return: a status (consistency, inconsistency or entailment) as an integer
   :rtype: int

