# Changelog

Notable changes to NuCS. This file starts at 15.0.0; for earlier releases see the
[git history](https://github.com/yangeorget/nucs/commits/main) and the release tags.

NuCS follows [semantic versioning](https://semver.org/): a major bump means the extension points
documented in [the docs](https://nucs.readthedocs.io/) changed shape.

## Unreleased

### Changed

- **`solve_one` is now `solve_one_step`.** The jitted search returns `None` both when the search is over
  and when the trail or the choice point stack runs out of room — both are caller-allocated and neither can
  grow inside `@njit`, so it stops, says which one filled up, and `BacktrackSolver` grows that array and
  calls it again. One step is what it does; the old name promised an outcome it cannot always deliver, and
  hid the reason the loop around it exists. It is not one of the extension points the docs cover, so the
  rename is not a major on its own, but the symbol is importable, so an import of it by name now fails:

  ```python
  from nucs.solvers.backtrack_solver import solve_one_step  # was solve_one
  ```

## 16.0.0

### Fixed

- **A propagator registered after import could not be solved with.** `IDEMPOTENCIES` was a Numpy array
  and `register_propagator` grew it with `np.append`, which returns a new array — so registering rebound
  the name, and the solver, which had imported it by value at module load, kept an array one entry short of
  every algorithm registered since. The new algorithm's id indexed past its end: an `IndexError` under
  `NUMBA_DISABLE_JIT=1`, and under the JIT a read past the array with `boundscheck` off, deciding
  idempotence from whatever followed it — whose wrong answer is the unsound one, leaving a non-idempotent
  propagator unrescheduled. `IDEMPOTENCIES` is now a `list[bool]` appended to in place, like the four
  registries beside it, and `Problem.init` builds the boolean array the consistency algorithm needs.

### Removed

- **`Problem.split`.** It partitioned a variable's domain into sub-problems for `MultiprocessingSolver`,
  which 15.0.0 removed; nothing has called it since, and splitting a problem is only useful when something
  then solves the parts independently.

### Changed

- **A consistency algorithm's `idempotent` parameter is now `idempotencies`, and moves to second place**,
  right after `statistics`. The plural matches what it is — one flag per algorithm, indexed by algorithm
  rather than by propagator. The module-level `IDEMPOTENT` is `IDEMPOTENCIES` to match, and is read through
  `get_idempotencies()`.
- **A consistency algorithm no longer takes `propagator_nb`**, and neither does `update_propagators`. The
  number was redundant with four of the arrays passed beside it — `len(priorities)`, `len(entailed)`,
  `len(algorithms)` and `len(offsets) - 1` — and `buckets_init` / `buckets_empty` already derived it
  rather than taking it. It is now derived from `priorities` at the one place that needs it, the
  `membership_offset` each queue operation is given.

  Migrating a custom consistency algorithm is deleting the first parameter, and a call to
  `update_propagators` is deleting the sixth argument:

  ```python
  # 15.0
  def my_consistency_algorithm(propagator_nb, statistics, algorithms, priorities, ...):
      update_propagators(triggered_propagators, entailed, triggers, triggers_offsets,
                         priorities, propagator_nb, variable, events)

  # now
  def my_consistency_algorithm(statistics, algorithms, priorities, ...):
      update_propagators(triggered_propagators, entailed, triggers, triggers_offsets,
                         priorities, variable, events)
  ```

  `Problem.propagator_nb` is unaffected: the attribute stays, it is only the threading of it through the
  jitted call chain that goes.

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
for both. It now only says *where* to split -- returning a kind and a value, mutating nothing -- and the solver
does the rest.

```python
# 14.x
def my_dom_heuristic(domains_stk, domain_update_stk, unbound_variable_nb_stk, stks_top, variable, params):
    ...  # mutate the stacks, return the events of the branch explored now

# 15.0
def my_dom_heuristic(domains: NDArray, variable: int, params: NDArray) -> tuple[int, int]:
    return DECISION_LE, (domains[variable, MIN] + domains[variable, MAX]) >> 1
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
- The `--processors` and `--cp-max-height` options of the examples' command line. `--processors` had
  nothing left to drive once the multiprocessing solvers went; `--cp-max-height` set a ceiling that is now
  a starting size the solver grows on its own, so raising it only reserved rows nothing ever touched.
  `BacktrackSolver(choice_point_max_height=...)` is unchanged for the rare caller who wants it.
- The `ALG_MAX_LEQ` and `ALG_MIN_GEQ` propagators.
- `split_random_dom_heuristic` and `DOM_HEURISTIC_SPLIT_RANDOM` — replaced by `random_value_dom_heuristic`
  / `DOM_HEURISTIC_RANDOM_VALUE`, which draws a value rather than a half and so is FlatZinc's
  `indomain_random`. The old one answered no annotation and was reachable from nothing.
- `value_dom_heuristic` (see above).
- Python 3.11: 3.12 or later is required.

### Changed

- **`BacktrackSolver.__init__`**: `stks_max_height` is now `choice_point_max_height`, and a
  `trail_max_size` was added. Both are starting sizes rather than ceilings — the solver grows either
  array and resumes rather than overrunning it, which also closes two missing overflow guards where
  exceeding the height used to write out of bounds under JIT. Both now default to `None`, which sizes
  them from the model: no benchmark model reaches even the flat floor, so growth only ever concerns
  models of a few thousand variables and up, which is where the model-derived floor takes over.
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
- FlatZinc: `indomain_random` is translated (to the new `random_value_dom_heuristic`), and a variable or
  value selector NuCS does not implement — `dom_w_deg`, `occurrence`, `outdomain_min`, … — now logs a
  warning naming what it was replaced with, instead of being silently solved as `input_order` /
  `indomain_min`.
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
