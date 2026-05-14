---
name: add-propagator
description: Scaffold a new constraint propagator in nucs/propagators/, register it, and add its test. Use when the user asks to add a propagator, implement a new constraint, or extend the propagator library.
---

# Add a propagator

A propagator is a constraint enforced by domain filtering. Adding one means: writing a module with three Numba-jitted
functions, registering it with an `ALG_*` id, and adding a parameterized test.

## 1. Pick a name and signature

- `<name>` is snake_case, derived from the constraint (e.g. `abs_eq`, `sum_leq_c`). Suffix `_c` means a constant
  parameter is involved.
- Decide what `domains` and `parameters` carry:
    - `domains` is an `NDArray` of shape `(n, 2)` — one `(MIN, MAX)` row per variable, in a fixed order chosen by you.
    - `parameters` is a 1-D `NDArray` of ints (may be empty). Use it for constants, coefficients, or table data.
- Document the variable order in the `compute_domains_<name>` docstring — callers rely on it.

## 2. Create `nucs/propagators/<name>_propagator.py`

The file must contain three functions and the standard copyright header (copy from `header.txt` — every source file in
`nucs/` and `tests/` starts with the ASCII-art banner).

Reference: `nucs/propagators/abs_eq_propagator.py` is the minimal template.

```python
def get_complexity_<


name > (n: int, parameters: NDArray) -> int:
# Not jitted. Return an int estimate of work per call.
# Used to order propagators in the queue — relative magnitude matters, not units.
...


@njit(cache=True, fastmath=True)
def get_triggers_<


name > (n: int, variable: int, parameters: NDArray) -> int:
# Return an EVENT_MASK_* constant from nucs.constants for the given variable index.
# Controls when this propagator wakes up after another propagator filters that variable.
...


@njit(cache=True, fastmath=True)
def compute_domains_<


name > (domains: NDArray, parameters: NDArray) -> int:
# Mutate domains in place. Return PROP_INCONSISTENCY, PROP_CONSISTENCY, or PROP_ENTAILMENT.
# Use domains[i][MIN] and domains[i][MAX]; never reassign domains[i] = ....
...
```

Rules for the jitted functions (see [[numba-rules]] in CLAUDE.md):

- No Python objects, no exceptions, no list/dict comprehensions over heterogeneous types.
- Mutate `domains` in place. After each tightening, check
  `if domains[i][MIN] > domains[i][MAX]: return PROP_INCONSISTENCY`.
- Return `PROP_ENTAILMENT` only when the constraint can never be violated again (rare; safe to return `PROP_CONSISTENCY`
  if unsure).

## 3. Register in `nucs/propagators/propagators.py`

Add the import alongside the others, then append a registration line. The `ALG_*` lines are ordered alphabetically by
id — keep that.

```python
from nucs.propagators. < name > _propagator
import

(
    compute_domains_ < name >,
    get_complexity_ < name >,
    get_triggers_ < name >,
)
...
ALG_ < NAME > = register_propagator(get_triggers_ < name >, get_complexity_ < name >, compute_domains_ < name >)
```

The returned id is the propagator's index; never hardcode it.

## 4. Add `tests/propagators/test_<name>.py`

Follow the `PropagatorTest` pattern (see `tests/propagators/test_abs_eq.py`):

```python
class Test< Name > (PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(lo, hi), ...], [param, ...], PROP_CONSISTENCY, [[lo, hi], ...]),
            # one row per case: boundary, inconsistency, entailment, no-change
        ],
    )
    def test_compute_domains(self, domains, parameters, consistency_result, expected_domains) -> None:
        self.assert_compute_domains(
            compute_domains_ < name >, domains, parameters, consistency_result, expected_domains
        )
```

Cover at minimum: a pruning case, an inconsistency case, and a no-op case where the input is already tight.

## 5. Verify

```bash
./scripts/bash/style.sh
NUMBA_CACHE_DIR=.numba/cache PYTHONPATH=. pytest tests/propagators/test_<name>.py
```

For debugging the propagator logic interactively, run with `NUMBA_DISABLE_JIT=1` so tracebacks land in your Python
source.

## 6. Document

Add the propagator to docs/source/reference_propagators.rst.