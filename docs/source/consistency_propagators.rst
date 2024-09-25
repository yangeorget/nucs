###########################
Consistency and propagators
###########################

***********
Consistency
***********

Consistency algorithms
######################
NUCS implements bound consistency out-of-the box and supports custom consistency algorithms.

*****************************
Propagators (aka constraints)
*****************************

NUCS comes with some highly-optimized :ref:`propagators <propagators>`.
Each propagator :code:`XXX` defines three functions:

- :code:`compute_domains_XXX(domains: NDArray, data: NDArray) -> int`
- :code:`get_triggers_XXX(size: int, data: NDArray) -> NDArray`
- :code:`get_complexity_XXX(size: int, data: NDArray) -> float`

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
:code:`triggers[i, MIN] == True` means that the propagator should be triggered whenever the minimum value of variable :code:`ì` changes.

:code:`get_complexity` function
###############################

This function returns the amortized complexity of the propagator's :code:`compute_domains` method as a :code:`float`.

These complexities are used to sort the propagators and ensure that the cheapest propagators are evaluated first.


