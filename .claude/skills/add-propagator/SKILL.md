---
name: add-propagator
description: Skill to add a propagator to NuCS.
---

# Add a propagator

A propagator is a constraint enforced by domain filtering.
Adding one means:

- writing a module with three Numba-jitted functions
- registering it with an `ALG_*` id
- adding a parameterized test

## 1. Pick a name and signature

- `name` is snake_case, derived from the constraint (e.g. `abs_eq`, `sum_leq_c`). Suffix `_c` means a constant
  parameter is involved.
- Decide what `domains` and `parameters` carry:
    - `domains` is an `NDArray` of shape `(n, 2)` — one `(MIN, MAX)` row per variable, in a fixed order chosen by you.
    - `parameters` is a 1-D `NDArray` of ints (may be empty). Use it for constants, coefficients, or table data.
- Document the variable order in the `compute_domains_name` docstring — callers rely on it.

## 2. Create `nucs/propagators/name_propagator.py`

The file must contain three functions and the standard copyright header (use the skill add-header).

Reference: `nucs/propagators/abs_eq_propagator.py` is the minimal template.

```python
def get_complexity_name(n: int, parameters: NDArray) -> int:
    # Not jitted. Return an int estimate of work per call.
    # Used to order propagators in the queue — relative magnitude matters, not units.
    ...


@njit(cache=True)
def get_triggers_name(n: int, variable: int, parameters: NDArray) -> int:
    # Return an EVENT_MASK_* constant from nucs.constants for the given variable index.
    # Controls when this propagator wakes up after another propagator filters that variable.
    ...


@njit(cache=True)
def compute_domains_name(domains: NDArray, parameters: NDArray) -> int:
    # Mutate domains in place. Return PROP_INCONSISTENCY, PROP_CONSISTENCY, or PROP_ENTAILMENT.
    # Use domains[i][MIN] and domains[i][MAX]; never reassign domains[i] = ....
    # If one call cannot reach this propagator's fixpoint, register it with idempotent=False — see below.
    ...
```

Rules for the jitted functions:

- Use the skill write-numba-friendly-python-code.
- No Python objects, no exceptions, no list/dict comprehensions over heterogeneous types.
- Mutate `domains` in place.
  After each tightening, check `if domains[i][MIN] > domains[i][MAX]: return PROP_INCONSISTENCY`.
- Return `PROP_ENTAILMENT` only when the constraint can never be violated again
  (rare; safe to return `PROP_CONSISTENCY` if unsure).
- **Decide whether `compute_domains` is idempotent, and declare it.** Idempotent means one call reaches the
  propagator's own fixpoint: a second consecutive call changes nothing. The engine never reschedules a
  propagator onto its own prunes (the `other_prop_idx == prop_idx` skip in `bc_algorithm.py`), so it will not
  re-run you to finish the job unless you say it must.

  If one pass can leave filtering on the table — tightening one variable re-opens filtering for another the
  pass already visited, as in any pairwise or cascading rule — register with `idempotent=False` and the engine
  puts you back on the propagation queue after any call that changed a domain, until a call changes nothing:

  ```python
  ALG_NAME = register_propagator(
      get_triggers_name, get_complexity_name, compute_domains_name, idempotent=False
  )
  ```

  Going back through the queue rather than looping internally lets cheaper propagators — and any inconsistency
  they expose — run in between. See `cumulative`, `diffn`, `disjunctive`, `linear_eq_c` for propagators that
  declare it.

  **Getting this wrong in the `True` direction is a soundness bug, not a filtering weakness.** If your own
  pruning can *create* an inconsistency that the same pass then fails to detect, and nothing re-runs you, an
  infeasible assignment reaches a leaf and is reported as a solution — this is exactly what `diffn` did. The
  default is `idempotent=True`, i.e. the dangerous side, so do not leave it implicit: **verify it.** Call
  `compute_domains` twice on random contract-valid instances and assert the second call changes nothing.
  Purely functional propagators (each variable filtered once from fixed data) are idempotent by construction.

## 3. Register in `nucs/propagators/propagators.py`

Add the import alongside the others, then append a registration line.
The `ALG_*` lines are ordered alphabetically by id — keep that.

```python
from nucs.propagators.name_propagator import compute_domains_name, get_complexity_name, get_triggers_name

...
ALG_NAME = register_propagator(get_triggers_name, get_complexity_name, compute_domains_name)
```

The returned id is the propagator's index; never hardcode it.

## 4. Add `tests/propagators/test_name.py`

Follow the `PropagatorTest` pattern (see `tests/propagators/test_abs_eq.py`):

```python
class TestName(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(lo, hi), ...], [param, ...], PROP_CONSISTENCY, [[lo, hi], ...]),
            # one row per case: boundary, inconsistency, entailment, no-change
        ],
    )
    def test_compute_domains(self, domains, parameters, consistency_result, expected_domains) -> None:
        self.assert_compute_domains(compute_domains_name, domains, parameters, consistency_result, expected_domains)
```

Cover at minimum: a pruning case, an inconsistency case, and a no-op case where the input is already tight.

Also guard the two invariants above:

- **Idempotence:** feeding a `compute_domains` result back into a second call must change nothing — the
  expected domains of every pruning case are themselves a fixpoint, so `assert_compute_domains(fct, expected,
  parameters, PROP_*, expected)` must hold. For any propagator with pairwise or cascading rules, add a
  brute-force soundness test (see `tests/propagators/test_diffn.py::test_soundness_against_brute_force`): over
  small enumerated domains, assert the propagator never reports consistent a state that has no feasible ground
  extension, and never prunes a value that belongs to a solution.

## 5. Verify

```bash
./scripts/bash/style.sh
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/propagators/test_name.py
```

For debugging the propagator logic interactively,
run with `NUMBA_DISABLE_JIT=1` so tracebacks land in your Python source.

## 6. Document

Add the propagator to `docs/source/reference/reference_propagators.rst`
(the `.. autofunction::` list is ordered alphabetically by module name — keep that).

## 7. Wire it into the FlatZinc adapter (only if it backs a FlatZinc builtin)

If the propagator exists to support a MiniZinc/FlatZinc constraint, also connect it in `nucs/fzn/`:

- Add one entry to the `BUILTINS` dict in `nucs/fzn/builtins.py`, keyed by the FlatZinc builtin name. Its handler
  resolves the args (`model.var_index_of` / `var_list_of` / `int_list_of` / `const_of`) and calls
  `model.problem.add_propagator(ALG_NAME, variables, parameters)`.
- If it is a **global** you want MiniZinc to keep native rather than decompose, add a body-less predicate file under
  `nucs/fzn/share/minizinc/nucs/` (see `fzn_all_different_int.mzn`), and key the dispatch entry on the `fzn_*` (or
  custom)
  predicate name that file produces.

Verify end-to-end with `tests/fzn/` and, when `minizinc` is installed, `minizinc --solver nucs`.