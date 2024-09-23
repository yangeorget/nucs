API
===

.. autosummary::
   :toctree: generated

   lumache



### Consistency

#### Consistency algorithms
NUCS implements bound consistency out-of-the box and supports custom consistency algorithms.

#### Propagators (aka constraints)
Each propagator `XXX` defines three functions:
- `compute_domains_XXX(domains: NDArray, data: NDArray) -> int`
- `get_triggers_XXX(size: int, data: NDArray) -> NDArray`
- `get_complexity_XXX(size: int, data: NDArray) -> float`

##### `compute_domains`
This function takes as its first argument the actual domains (not the shared ones) of the variables of the propagator
and updates them.

It is expected to implement bound consistency and to be idempotent
(a second consecutive run should not update the domains).

It returns a status:
- `PROP_INCONSISTENCY`,
- `PROP_CONSISTENCY` or
- `PROP_ENTAILMENT`.

##### `get_triggers`
This function returns a `numpy.ndarray` of shape `(size, 2)`.

Let `triggers` be such an array,
`triggers[i, MIN] == True` means that the propagator should be triggered whenever the minimum value of variable `ì` changes.

##### `get_complexity`
This function returns the amortized complexity of the propagator's `compute_domains` method as a `float`.

These complexities are used to sort the propagators and ensure that the cheapest propagators are evaluated first.


