---
name: add-propagator
description: Adds a constraint propagator to NuCS end to end — the propagator module, its ALG_* registration, tests, reference docs and, when needed, the FlatZinc builtin. Use when implementing a new constraint or global, when a FlatZinc/MiniZinc builtin needs a native propagator, or when giving an existing propagator a vacuity check, a state block or change reporting.
---

# Add a propagator

A propagator enforces one constraint by bounds filtering. Copy this checklist and tick it off:

```
- [ ] 1. Fix the name and the domains/parameters contract
- [ ] 2. Write nucs/propagators/<name>_propagator.py
- [ ] 3. Register ALG_<NAME> in nucs/propagators/propagators.py
- [ ] 4. Write tests/propagators/test_<name>.py
- [ ] 5. Add the autofunction to docs/source/reference/reference_propagators.rst
- [ ] 6. Wire nucs/fzn/ (only if it backs a FlatZinc builtin)
- [ ] 7. Style and tests pass
```

## 1. Name and contract

- `<name>` is snake_case after the constraint (`abs_eq`, `sum_leq_c`); a `_c` suffix means a constant parameter.
- `domains` is an int32 array of shape `(n, 2)`: one `[DOMAIN_MIN, DOMAIN_MAX]` row per variable, in an order you
  choose. Document that order in the `compute_domains_<name>` docstring — callers depend on it.
- `parameters` is a 1-D int32 array, possibly empty: constants, coefficients, table data.

## 2. Write the module

Start from `nucs/propagators/abs_eq_propagator.py`, the minimal template. The file starts with the banner from
`header.txt` (the add-header skill), docstrings follow the write-docstring skill, and the jitted functions follow the
write-numba-friendly-python-code skill.

```python
def get_complexity_<name>(n: int, parameters: NDArray) -> int:
    # plain Python: a work estimate that orders the propagation queue; only relative magnitude matters


@njit(cache=True)
def get_triggers_<name>(n: int, variable: int, parameters: NDArray) -> int:
    # the EVENT_MASK_* from nucs.constants that wakes this propagator when `variable` is narrowed


@njit(cache=True)
def compute_domains_<name>(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    # narrow domains in place; return PROP_INCONSISTENCY, PROP_CONSISTENCY or PROP_ENTAILMENT
```

Rules for `compute_domains_<name>`:

- Take `prop_state` even when unused: every `compute_domains` is compiled against the same three-argument signature.
- Write bounds in place (`domains[i, DOMAIN_MIN] = ...`, or through a row view `x = domains[i]`); never rebind a row.
- After tightening a variable, return `PROP_INCONSISTENCY` as soon as its `DOMAIN_MIN` exceeds its `DOMAIN_MAX`.
- Return `PROP_ENTAILMENT` only when no later narrowing can violate the constraint. `PROP_CONSISTENCY` is always safe.
- Do not allocate: an `np.empty`/`np.zeros` here is paid at every fixpoint. Take scratch space from a state block.

### Decide idempotence (every propagator)

A `compute_domains` is idempotent when a second consecutive call changes nothing. The engine never wakes a
propagator on its own prunes (the `other_prop_idx == prop_idx` skip in `nucs/solvers/bc_algorithm.py`), so if one
pass can leave filtering undone — pairwise or cascading rules, where tightening one variable reopens one already
visited — register with `idempotent=False`. The engine then requeues the propagator after every call that changed a
domain, letting cheaper propagators run in between. Examples: `cumulative`, `diffn`, `disjunctive`, `linear_eq_c`,
`regular`.

The default, `idempotent=True`, is the unsafe direction. If a pass creates an inconsistency it fails to detect and
nothing reruns it, an infeasible assignment is reported as a solution — the `diffn` bug. Verify instead of assuming:
call `compute_domains` twice on random valid inputs and assert the second call changes nothing. A propagator that
filters each variable once from fixed data is idempotent by construction.

### Optional declarations

Read the reference only when its row applies:

| Declaration             | Add it when                                                                                                     | Reference                  |
|-------------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------|
| `is_vacuous_<name>`     | the parameters, or the parameters and initial domains, can make the constraint unviolable — common in FlatZinc | `references/vacuity.md`     |
| `get_state_<name>`      | the propagator needs scratch space, a hint carried across calls, or trailed per-node state                     | `references/state-block.md` |
| `reports_changes=True`  | calls often narrow nothing, so the engine should skip its write-back scan                                      | `references/state-block.md` |

## 3. Register

In `nucs/propagators/propagators.py`, import the functions and add the registration in alphabetical position among
the `ALG_*` lines:

```python
ALG_<NAME> = register_propagator(get_triggers_<name>, get_complexity_<name>, compute_domains_<name>)
```

The optional declarations are the remaining parameters; each one omitted takes its default:

```python
ALG_<NAME> = register_propagator(
    get_triggers_<name>,
    get_complexity_<name>,
    compute_domains_<name>,
    is_vacuous_<name>,               # default is_never_vacuous: always post
    idempotent=False,                # default True: verify it, see above
    get_state_fct=get_state_<name>,  # default get_state_default: no state block
    reports_changes=True,            # default False
)
```

The return value indexes the registry lists and `ALGORITHM_FLAGS`; never hardcode it. The registries are lists
appended in place so that a propagator registered after import is visible to modules that already imported them.

## 4. Test

Create `tests/propagators/test_<name>.py` on the `PropagatorTest` pattern of `tests/propagators/test_abs_eq.py`:

```python
class TestName(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(lo, hi), ...], [param, ...], PROP_CONSISTENCY, [[lo, hi], ...]),
        ],
    )
    def test_compute_domains(self, domains, parameters, consistency_result, expected_domains) -> None:
        self.assert_compute_domains(compute_domains_<name>, domains, parameters, consistency_result, expected_domains)
```

`assert_compute_domains` sizes a zeroed state block from `get_state_<name>`, iterates a non-idempotent propagator to
its fixpoint as the engine does, and fails a `reports_changes` propagator that reports no change but narrowed a domain.

Cover at least a pruning case, an inconsistency case, a case already at its fixpoint, and entailment if you return it.
Then guard idempotence:

- For an idempotent propagator, the expected domains of each pruning case are a fixpoint, so
  `assert_compute_domains(fct, expected, parameters, status, expected)` must hold.
- For pairwise or cascading rules, add a brute-force soundness test like
  `tests/propagators/test_diffn.py::test_soundness_against_brute_force`: over small enumerated domains, the
  propagator never reports consistent a state with no feasible ground extension, and never prunes a value that
  belongs to a solution.

Vacuity and state blocks have their own tests, described in their references.

## 5. Document

Add `.. autofunction:: nucs.propagators.<name>_propagator.compute_domains_<name>` to
`docs/source/reference/reference_propagators.rst`, in alphabetical position.

## 6. Wire FlatZinc (only if it backs a builtin)

- Add an entry to `BUILTINS` in `nucs/fzn/builtins.py`, keyed by the FlatZinc builtin name. Its handler takes
  `(model, args)`, resolves the args with `model.var_index_of` / `var_list_of` / `int_list_of` / `const_of`, and calls
  `model.problem.add_propagator(ALG_<NAME>, variables, parameters)`.
- For a global MiniZinc should keep native rather than decompose, add a body-less predicate under
  `nucs/fzn/share/minizinc/nucs/` (see `fzn_all_different_int.mzn`) and key the `BUILTINS` entry on the predicate
  name that file produces.

## 7. Verify

```bash
./scripts/bash/style.sh
NUMBA_CACHE_DIR=.numba/cache pytest tests/propagators/test_<name>.py
NUMBA_CACHE_DIR=.numba/cache pytest tests/fzn  # if step 6 applied
```

Fix and rerun until everything passes. For a cryptic Numba error, rerun with `NUMBA_DISABLE_JIT=1` to get a traceback
on the real source line. `tests/fzn/test_minizinc.py` runs the *installed* `fzn-nucs` through MiniZinc, so reinstall
first: `rm -rf build && pip install --no-deps .`.
