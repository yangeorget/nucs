###########################
Consistency and propagators
###########################

***********
Consistency
***********

Consistency algorithms
######################
NuCS implements bound consistency out-of-the box and supports custom consistency algorithms.

Bound consistency
#################
Unless specified otherwise, bound consistency is used.

.. py:module:: nucs.solvers.bound_consistency_algorithm
.. py:function:: nucs.solvers.bound_consistency_algorithm.bound_consistency_algorithm(statistics, algorithms, var_bounds, param_bounds, dom_indices_arr, dom_offsets_arr, props_dom_indices, props_dom_offsets, props_parameters,shr_domains_propagators, shr_domains_arr, not_entailed_propagators, triggered_propagators, compute_domains_addrs)

   :param statistics: a Numpy array of statistics
   :type statistics: NDArray
   :param algorithms: the algorithms indexed by propagators
   :type algorithms: NDArray
   :param var_bounds: the variable bounds indexed by propagators
   :type var_bounds: NDArray
   :param param_bounds: the parameters bounds indexed by propagators
   :type param_bounds: NDArray
   :param dom_indices_arr: the domain indices indexed by variables
   :type dom_indices_arr: NDArray
   :param dom_offsets_arr: the domain offsets indexed by variables
   :type dom_offsets_arr: NDArray
   :param props_dom_indices: the domain indices indexed by propagator variables
   :type props_dom_indices: NDArray
   :param props_dom_offsets: the domain offsets indexed by propagator variables
   :type props_dom_offsets: NDArray
   :param props_parameters: the parameters indexed by propagator variables
   :type props_parameters: NDArray
   :param shr_domains_propagators: a Numpy array of booleans indexed by shared domain indices, MIN/MAX and propagators; true means that the propagator has to be triggered when the MIN or MAX of the shared domain has changed
   :type shr_domains_propagators: NDArray
   :param shr_domains_arr: the current shared domains
   :type shr_domains_arr: NDArray
   :param not_entailed_propagators: the propagators currently not entailed
   :type not_entailed_propagators: NDArray
   :param triggered_propagators: the Numpy array of triggered propagators
   :type triggered_propagators: NDArray
   :param compute_domains_addrs: the addresses of the compute_domains functions
   :type compute_domains_addrs: NDArray
   :return: a status (consistency, inconsistency or entailment) as an integer
   :rtype: int

Custom consistency algorithms
#############################
NuCS makes it possible to use custom consistency algorithms.

The :mod:`nucs.examples.golomb.golomb_problem` model defines a custom consistency algorithm.

This custom consistency algorithm needs to be registered before it is used.

.. code-block:: python
   :linenos:

   consistency_alg_golomb = register_consistency_algorithm(golomb_consistency_algorithm)
   solver = BacktrackSolver(problem, consistency_alg_idx=consistency_alg_golomb)


*****************************
Propagators (aka constraints)
*****************************

NuCS comes with some highly-optimized :ref:`propagators <propagators>`.
Each propagator :code:`XXX` defines three functions:

- :code:`compute_domains_XXX(domains: NDArray, parameters: NDArray) -> int`
- :code:`get_triggers_XXX(size: int, parameters: NDArray) -> NDArray`
- :code:`get_complexity_XXX(size: int, parameters: NDArray) -> float`

:code:`compute_domains` function
################################

This function takes as its first argument the actual domains (not the shared ones) of the variables of the propagator
and updates them.

It is expected to implement bound consistency and to be idempotent
(a second consecutive run should not update the domains).

It returns a status:

- :code:`PROP_INCONSISTENCY`,
- :code:`PROP_CONSISTENCY` or
- :code:`PROP_ENTAILMENT`.

:code:`get_triggers` function
#############################

This function returns a :code:`numpy.ndarray` of shape :code:`(size, 2)`.

Let :code:`triggers` be such an array,
:code:`triggers[i, MIN] == True` means that
the propagator should be triggered whenever the minimum value of variable :code:`ì` changes.

:code:`get_complexity` function
###############################

This function returns the amortized complexity of the propagator's :code:`compute_domains` method as a :code:`float`.

These complexities are used to sort the propagators and ensure that the cheapest propagators are evaluated first.


