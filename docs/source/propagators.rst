#############################
Propagators (aka constraints)
#############################

NuCS comes with some highly-optimized :ref:`propagators <propagators>`.


********************
Propagator functions
********************

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

This function returns a :code:`numpy.ndarray` of event masks of shape :code:`size`.


:code:`get_complexity` function
###############################

This function returns the amortized complexity of the propagator's :code:`compute_domains` method as a :code:`float`.

These complexities are used to sort the propagators and ensure that the cheapest propagators are evaluated first.


******************
Custom propagators
******************
NuCS makes it possible to define and use custom propagators.

A propagator needs to be registered before it is used.
The following code registers the :code:`AND` propagator.

.. code-block:: python
   :linenos:

   ALG_AND = register_propagator(get_triggers_and, get_complexity_and, compute_domains_and)

