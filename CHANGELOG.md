# Changelog

Notable changes to NuCS. This file starts at 15.0.0; for earlier releases see the
[git history](https://github.com/yangeorget/nucs/commits/main) and the release tags.

NuCS follows [semantic versioning](https://semver.org/): a major bump means the extension points
documented in [the docs](https://nucs.readthedocs.io/) changed shape.

## 15.0.0

The headline is that backtrackable state is now **trailed rather than copied**, which removes the hard
ceiling on model size. Along the way both heuristic extension points and the consistency-algorithm
contract changed, so this is a major release.

### Migrating from 14.x

Four things can break a working 14.x program. The first three are compile-time failures; the fourth is
not, which is why it is worth reading.

#### 1. Domain heuristics are now pure decision functions

A domain heuristic used to perform the branching itself: push a choice point, write the explored branch
into one stack level and the parked alternative into another, and maintain the unbound-variable count
for both. It now only says *where* to split, and the solver does the rest.

```python
# 14.x
def my_dom_heuristic(domains_stk, domain_update_stk, unbound_variable_nb_stk, stks_top, variable, params):
    ...  # mutate the stacks, return the events of the branch explored now

# 15.0
def my_dom_heuristic(domains: NDArray, variable: int, params: NDArray, decision: NDArray) -> int:
    decision[DECISION_VALUE] = (domains[variable, MIN] + domains[variable, MAX]) >> 1
    return DECISION_LE
```

Return one of three kinds; the solver turns it into an explored branch and one or two parked
alternatives, resumed in the order listed:

| kind | explored now | parked (deepest first) |
|------|--------------|------------------------|
| `DECISION_LE` at `v` | `[min, v]` | `[v + 1, max]` |
| `DECISION_GT` at `v` | `[v + 1, max]` | `[min, v]` |
| `DECISION_EQ` at `v` | `[v, v]` | `[min, v - 1]` then `[v + 1, max]` |

`value_dom_heuristic` is deleted: it was the `DECISION_EQ` case, which the solver now handles.

#### 2. Variable heuristics receive the domains, not the stack

```python
# 14.x
def my_var_heuristic(decision_variables, domains_stk, top, params): ...
# 15.0
def my_var_heuristic(decision_variables: NDArray, domains: NDArray, params: NDArray) -> int: ...
```

`domains` is a `(domain_nb, 2)` array indexed by variable then `MIN` / `MAX`. Mechanical: replace
`domains_stk[top, variable]` with `domains[variable]`.

#### 3. Consistency algorithms receive one set of domains, behind a write barrier

`SIGN_CONSISTENCY_ALG` no longer hands out a stack of domains. **This is a semantic change, not only a
shape change**, and it is the one that can fail quietly.

A custom consistency algorithm must write domains through
`nucs.solvers.choice_points.tighten` (or `tighten_at` in a loop). Assigning a domain directly still
compiles and still looks right, but the write is not recorded on the trail: it is never undone, and it
survives the backtrack into sibling subtrees. There is no error to point at.

```python
events = tighten(state, trail_log, trail_top, trail_indices, mark, variable, new_min, new_max)
if events:
    update_propagators(triggered_propagators, entailed, triggers, triggers_offsets,
                       priorities, propagator_nb, variable, events)
```

where `mark` is `choice_point_stk[choice_point_top[0], CHOICE_POINT_TRAIL_MARK]`. `tighten` also owns
the groundness test and the unbound-variable count, so routing through it is what keeps the solver's
"is this problem solved?" test correct. `nucs/examples/golomb/golomb_problem.py` is the in-tree example.

Propagators are unaffected: `SIGN_COMPUTE_DOMAINS` is unchanged, and a `compute_domains_*` function
still receives a gathered copy of only its own variables' domains.

#### 4. Custom propagators must be idempotent, or say they are not

A propagator is expected to reach its own fixpoint in a single call. One that does not must be
registered as such: the engine never reschedules a propagator onto its own prunes, so one that stops
short of its own fixpoint is never re-run and can accept an assignment it would itself reject.

```python
ALG_MINE = register_propagator(get_triggers_mine, get_complexity_mine, compute_domains_mine,
                               idempotent=False)
```

This is not a compile-time failure. A non-idempotent propagator registered as idempotent produces wrong
answers, not errors.

### Removed

- `BacktrackSolver.minimize` and `.maximize` — use `find_best(variable, MIN, mode)` / `find_best(variable, MAX, mode)`,
  or iterate `optimize(...)`.
- `MultiprocessingSolver` and `QueueSolver`.
- The `ALG_MAX_LEQ` and `ALG_MIN_GEQ` propagators.
- `value_dom_heuristic` (see above).
- Python 3.11: 3.12 or later is required.

### Changed

- **`BacktrackSolver.__init__`**: `stks_max_height` is now `choice_point_max_height`, and a
  `trail_max_size` was added. Both are starting sizes rather than ceilings — the solver grows either
  array and resumes rather than overrunning it, which also closes two missing overflow guards where
  exceeding the height used to write out of bounds under JIT.
- **`Solver`** is a real abstract base class, and `solve`, `solve_all`, `find_all`, `optimize`,
  `optimize_all` and `find_best` all take an optional `timeout` in seconds. Under a timeout the search
  keeps its best result and sets `solver.timed_out`.
- **`get_statistics_as_dictionary()`** additionally breaks the two filtering counters down per
  propagator algorithm, for the algorithms that ran at least once.
- **`min_cost_dom_heuristic`** falls back to the domain's minimum when no value in the domain has a
  positive cost. It previously wrote an out-of-domain value and widened the alternative.
- **Memory and time.** Choice points no longer preallocate `stks_max_height × domain_nb × 8` bytes:
  `bibd(10,15,6,4,2)` goes from 51.7 MB to 0.6 MB and `magic_sequence(200)` from 12.6 MB to 0.6 MB.
  The search is unchanged — solutions, backtracks, BC calls, propagator calls and entailments are
  identical to 14.1.0 on all 27 benchmark models — and throughput costs about 6% overall, worst
  measured 12%. `ARCHITECTURE.md` records why that trade was taken.

### Added

- Native propagators: `ALG_REGULAR` (DFA), `ALG_BIN_PACKING_LOAD` with subset-sum filtering,
  `ALG_CUMULATIVE_VAR` (cumulative with variable durations), `ALG_IF_THEN_ELSE`, `ALG_DIV_C_EQ`,
  `ALG_NEQ_C_REIF`, `ALG_MEMBER_REIF`, and the half-reified `ALG_EQ_IMP`, `ALG_EQ_C_IMP`,
  `ALG_LEQ_C_IMP` and `ALG_NEQ_IMP`.
- FlatZinc: those globals stay native instead of falling back to a decomposition — variable-duration
  `cumulative` alone replaced 249 decomposed constraints on one model. Also `int_lin_le_imp`, cheaper
  propagators for common shapes, always-equal variables collapsed to a single NuCS variable, and the
  solver owns the time limit.
- `scripts/benchmark.py` reports, per model, the bytes preallocated for backtrackable state and the
  peak RSS of the process that solved it.

### Fixed

- `OPTIM_PRUNE` with a three-way domain heuristic and `MIN` hung the solver. The branch-and-bound bound
  is solver state rather than choice-point state, and is re-applied to each choice point as the search
  resumes it instead of being written into all of them up front. Reachable from the FlatZinc runner,
  which uses `OPTIM_PRUNE` and maps `indomain_median` to `DOM_HEURISTIC_MID_VALUE`.
- `diffn` could report overlapping solutions: the pairwise pass now iterates to a fixpoint.
- Two globals silently returned UNSAT for non-1-based index sets.
- Several FlatZinc interface violations, an alias-domain bug, and unsupported floats now reported as a
  type error rather than an unexpected character.
