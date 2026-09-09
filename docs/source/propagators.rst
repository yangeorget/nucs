#############################
Propagators (aka constraints)
#############################

NuCS comes with some highly-optimized :ref:`propagators <propagators>`.


********************
Propagator functions
********************

Each propagator :code:`XXX` defines three functions, and optionally two more:

- :code:`compute_domains_XXX(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int`
- :code:`get_triggers_XXX(n: int, variable: int,  parameters: NDArray) -> int`
- :code:`get_complexity_XXX(size: int, parameters: NDArray) -> int`
- :code:`is_vacuous_XXX(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool`
- :code:`get_state_XXX(n: int, parameters: Sequence[int]) -> tuple[int, int]`


:code:`compute_domains` function
################################

This function takes as its first argument the domains of the variables of the propagator and updates them.

Its third argument, :code:`prop_state`, is this propagator's own slice of solver-owned :code:`int32` memory,
sized by its :code:`get_state` function and empty by default.
It replaces the per-call :code:`np.empty`/:code:`np.zeros` a propagator would otherwise allocate for scratch
space, and lets one call carry a hint over to the next.

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


:code:`get_state` function
##########################

This optional function returns :code:`(trailed_nb, hint_nb)`: how many :code:`int32` cells this propagator
wants in the state array, split into a backtrackable prefix and an untrailed suffix.
:code:`compute_domains` receives the two as one contiguous :code:`prop_state` block, the trailed cells first.

The solver saves and restores the :code:`trailed_nb` prefix like a domain bound, so it holds per-node state.
The :code:`hint_nb` suffix is never trailed: it keeps whatever the last call to *any* node left in it.
A propagator may only use it for values that are either fully overwritten before being read, or valid however
stale - a hint whose staleness costs time, never correctness.
:code:`alldifferent` uses it for both: its scratch arrays are overwritten on every call, and its warm-started
sort permutations stay valid because a stale permutation is still a permutation.

The whole block is zeroed once, at solver init, and *not* re-zeroed on backtrack or on an
:code:`OPTIM_RESET` restart - a propagator that needs a cleared block must clear it itself.

A propagator that does not define this function gets a zero-width block.


******************
Custom propagators
******************
NuCS makes it possible to define and use custom propagators.

A propagator needs to be registered before it is used.
The following code registers the :code:`AND` propagator.

.. code-block:: python
   :linenos:

   ALG_AND = register_propagator(get_triggers_and, get_complexity_and, compute_domains_and)

:code:`register_propagator` takes three further arguments: the :code:`is_vacuous` function described above,
:code:`idempotent`, and the :code:`get_state` function described above.
They default to :code:`is_never_vacuous`, which always posts, to :code:`True`, and to
:code:`get_state_default`, which asks for no state.

.. code-block:: python
   :linenos:

   ALG_XXX = register_propagator(
       get_triggers_XXX,
       get_complexity_XXX,
       compute_domains_XXX,
       is_vacuous_XXX,
       idempotent=False,
       get_state_fct=get_state_XXX,
   )

