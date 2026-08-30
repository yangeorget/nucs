#############################
Propagators (aka constraints)
#############################

NuCS comes with some highly-optimized :ref:`propagators <propagators>`.


********************
Propagator functions
********************

Each propagator :code:`XXX` defines three functions, and optionally a fourth:

- :code:`compute_domains_XXX(domains: NDArray, parameters: NDArray) -> int`
- :code:`get_triggers_XXX(n: int, variable: int,  parameters: NDArray) -> int`
- :code:`get_complexity_XXX(size: int, parameters: NDArray) -> int`
- :code:`is_vacuous_XXX(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool`


:code:`compute_domains` function
################################

This function takes as its first argument the domains of the variables of the propagator and updates them.

It is expected to implement bound consistency.

It should also be idempotent - a second consecutive run should not update the domains - since the
consistency algorithm never reschedules a propagator onto its own prunes.
A propagator that cannot reach its fixpoint in a single call must be registered with
:code:`idempotent=False`, in which case the engine puts it back on the propagation queue after every call
that changed a domain, until a call changes nothing.

It returns a status:

- :code:`PROP_INCONSISTENCY`,
- :code:`PROP_CONSISTENCY` or
- :code:`PROP_ENTAILMENT`.


:code:`get_triggers` function
#############################

This function returns an event mask.


:code:`get_complexity` function
###############################

This function returns the amortized complexity of the propagator's :code:`compute_domains` method.

These complexities are used to sort the propagators and ensure that the cheapest propagators are evaluated first.


:code:`is_vacuous` function
###########################

This optional function tells, from the parameters and the initial domains, whether the constraint is
vacuous - whether every assignment those domains allow satisfies it.

:code:`Problem.add_propagator` calls it before posting, and does not post the propagator at all when it
returns :code:`True`: there is then no call at every fixpoint, no entry in the trigger buckets and no slot
in the propagator arrays. Constraints settled this way are common in generated models - a
:code:`cumulative` whose capacity already covers the sum of every demand, or a
:code:`global_cardinality_low_up` whose capacities do not bite.

Unlike the three functions above, it is not jitted: it runs once per :code:`add_propagator`, in plain Python.

Returning :code:`True` when the constraint can still be violated silently drops it, and the search then
reports assignments that are not solutions.
The domains it receives are the ones held when the propagator is posted; since domains only shrink during
the search, a property established on them holds throughout, which is what makes it safe to look at the
domains and not only at the parameters.

A propagator that does not define this function is always posted.


******************
Custom propagators
******************
NuCS makes it possible to define and use custom propagators.

A propagator needs to be registered before it is used.
The following code registers the :code:`AND` propagator.

.. code-block:: python
   :linenos:

   ALG_AND = register_propagator(get_triggers_and, get_complexity_and, compute_domains_and)

:code:`register_propagator` takes two further arguments: the :code:`is_vacuous` function described above,
and :code:`idempotent`.
They default to :code:`is_never_vacuous`, which always posts, and to :code:`True`.

.. code-block:: python
   :linenos:

   ALG_XXX = register_propagator(
       get_triggers_XXX, get_complexity_XXX, compute_domains_XXX, is_vacuous_XXX, idempotent=False
   )

