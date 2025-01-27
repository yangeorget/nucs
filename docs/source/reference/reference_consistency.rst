.. _consistency_algorithms:

**********************
Consistency algorithms
**********************

NuCS provides the following consistency algorithms.

.. py:module:: nucs.solvers.bound_consistency_algorithm
.. py:function:: nucs.solvers.bound_consistency_algorithm.bound_consistency_algorithm(statistics, algorithms, var_bounds, param_bounds, dom_indices_arr, dom_offsets_arr, props_dom_indices, props_dom_offsets, props_parameters, triggers, shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, triggered_propagators, compute_domains_addrs, decision_domains)

   This is the default consistency algorithm used by the :mod:`nucs.solvers.backtrack_solver`.

   :param statistics: a Numpy array of statistics
   :type statistics: NDArray
   :param no_offsets: true iff all offsets are equal to 0
   :type no_offsets: boolean
   :param algorithms: the algorithms indexed by propagators
   :type algorithms: NDArray
   :param bounds: the bounds indexed by propagators
   :type bounds: NDArray
   :param variables_arr: the domain indices indexed by variables, unused here
   :type variables_arr: NDArray
   :param offsets_arr: the domain offsets indexed by variables, unused here
   :type offsets_arr: NDArray
   :param props_variables: the domain indices indexed by propagator variables
   :type props_variables: NDArray
   :param props_offsets: the domain offsets indexed by propagator variables
   :type props_offsets: NDArray
   :param props_parameters: the parameters indexed by propagator variables
   :type props_parameters: NDArray
   :param triggers: a Numpy array of event masks indexed by shared domain indices
   :type triggers: NDArray
   :param domains_stk: a stack of shared domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: a stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: a stack of domain updates (domain index and bound)
   :type dom_update_stk: NDArray
   :param stks_top: the height of the stacks as a Numpy array
   :type stks_top: NDArray
   :param triggered_propagators: the Numpy array of triggered propagators
   :type triggered_propagators: NDArray
   :param compute_domains_addrs: the addresses of the compute_domains functions
   :type compute_domains_addrs: NDArray
   :param decision_domnains: the decision domains
   :type decision_domains: NDArray
   :return: a status (consistency, inconsistency or entailment) as an integer
   :rtype: int


.. py:module:: nucs.solvers.shaving_consistency_algorithm
.. py:function:: nucs.solvers.shaving_consistency_algorithm.shaving_consistency_algorithm(statistics, algorithms, var_bounds, param_bounds, dom_indices_arr, dom_offsets_arr, props_dom_indices, props_dom_offsets, props_parameters, triggers, shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, triggered_propagators, compute_domains_addrs, decision_domains)

   This algorithm reduces the need of searching by shaving the domains.

   :param statistics: a Numpy array of statistics
   :type statistics: NDArray
   :param no_offsets: true iff all offsets are equal to 0
   :type no_offsets: boolean
   :param algorithms: the algorithms indexed by propagators
   :type algorithms: NDArray
   :param bounds: the bounds indexed by propagators
   :type bounds: NDArray
   :param variables_arr: the domain indices indexed by variables, unused here
   :type variables_arr: NDArray
   :param offsets_arr: the domain offsets indexed by variables, unused here
   :type offsets_arr: NDArray
   :param props_variables: the domain indices indexed by propagator variables
   :type props_variables: NDArray
   :param props_offsets: the domain offsets indexed by propagator variables
   :type props_offsets: NDArray
   :param props_parameters: the parameters indexed by propagator variables
   :type props_parameters: NDArray
   :param triggers: a Numpy array of event masks indexed by shared domain indices
   :type triggers: NDArray
   :param domains_stk: a stack of shared domains
   :type domains_stk: NDArray
   :param not_entailed_propagators_stk: a stack of not entailed propagators
   :type not_entailed_propagators_stk: NDArray
   :param dom_update_stk: a stack of domain updates (domain index and bound)
   :type dom_update_stk: NDArray
   :param stks_top: the height of the stacks as a Numpy array
   :type stks_top: NDArray
   :param triggered_propagators: the Numpy array of triggered propagators
   :type triggered_propagators: NDArray
   :param compute_domains_addrs: the addresses of the compute_domains functions
   :type compute_domains_addrs: NDArray
   :param decision_domnains: the decision domains
   :type decision_domains: NDArray
   :return: a status (consistency, inconsistency or entailment) as an integer
   :rtype: int

