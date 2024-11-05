###########################
Consistency and propagators
###########################

**********************
Consistency algorithms
**********************

Bound consistency algorithm
###########################
NuCS implements :mod:`nucs.solvers.bound_consistency_algorithm` out-of-the box.

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


