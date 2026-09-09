# NuCS architecture

NuCS is a **Constraint Satisfaction Problem solver** that uses **Numba JIT** for performance:
pure Python, compiled at runtime. All solver state lives in flat, preallocated NumPy arrays so the hot
path runs in Numba nopython mode with no Python objects.

## Repository structure

- **`nucs/problems/`** — a `Problem` carries `domains` (one `(min, max)` per variable; bound when `min == max`) and a
  list of propagators added via `add_propagator(ALG_*, *variable_index_iterables, parameters=...)`. `Problem.init()`
  flattens everything into the arrays the solver consumes (see *Data-oriented state* below).
- **`nucs/propagators/`** — one file per constraint, plus `propagators.py` which registers each as a numeric `ALG_*` id.
  Each propagator is three functions: `compute_domains_*` (filtering, returns `PROP_INCONSISTENCY` /
  `PROP_CONSISTENCY` / `PROP_ENTAILMENT`), `get_triggers_*` (when to re-wake), `get_complexity_*` (queue ordering). See
  `nucs/propagators/abs_eq_propagator.py` for the minimal template.
- **`nucs/solvers/`** — `BacktrackSolver` (backtracking + propagation). The propagation fixpoint is `bc_algorithm`
  (`bc_algorithm.py`), registered as `CONSISTENCY_ALG_BC`; the search driver is `solve_one_step`. The backtrackable
  state and its trail live in `state.py` — `tighten`/`tighten_at` are the only sanctioned way to write a domain — and
  the choice-point stack built on them lives in `choice_points.py`. Iterate solutions with `solver.solve()`, or
  optimize with `solver.find_best(var, DOMAIN_MIN)` / `solver.find_best(var, DOMAIN_MAX)`.
- **`nucs/heuristics/`** — variable heuristics pick the next unbound decision variable, domain heuristics pick how to
  split its domain. Both are Numba-jitted against the fixed signatures `SIGN_VAR_HEURISTIC` / `SIGN_DOM_HEURISTIC` in
  `nucs/heuristics/heuristics.py` and dispatched by id.
- **`nucs/fzn/`** — the **FlatZinc adapter**: model in MiniZinc, solve with NuCS via `minizinc --solver nucs`. Pipeline
  is `parser.py` (FlatZinc text → IR) → `model.py` (`FznModel` builds a `Problem`) → `builtins.py` (the `BUILTINS`
  dispatch table: FlatZinc builtin name → `add_propagator` calls) → `runner.py` (solve) → `output.py` (FlatZinc solution
  stream). `fzn-nucs` is the console script MiniZinc invokes; `fzn-nucs --register` writes the solver config into
  `~/.minizinc/solvers`. `share/minizinc/nucs/` is the globals library that keeps selected globals (alldifferent, gcc,
  lex, table) native instead of decomposed. Grow coverage by adding one entry to `BUILTINS` (and, for a kept global, one
  predicate file under `share/minizinc/nucs/`) — see the `/add-propagator` skill, step 7.

### The solve loop

`solve_one_step` drives the search for one solution, looping over two phases:

1. **Propagate to a fixpoint** (`bc_algorithm`): pop the cheapest triggered propagator from the queue, gather its
   variables' domains into `domain_buffer`, call its `compute_domains_*`, write back any tightened bounds, and enqueue
   the propagators the resulting events trigger. Repeat until the queue drains or a domain wipes out.
2. **React to the fixpoint status**:

   | status | meaning | action |
   |--------|---------|--------|
   | `PROBLEM_BOUND` | fixpoint reached, all variables bound | emit the solution |
   | `PROBLEM_INCONSISTENT` | a domain wiped out | `backtrack`: pop a choice point, replay the undo log back to its mark, reschedule the refuted decision. When optimizing, it keeps popping while the objective bound wipes the resumed one out |
   | `PROBLEM_UNBOUND` | fixpoint reached, unbound variables remain | `branch`: the first search with an unbound decision variable picks one (variable heuristic) and says where to split its domain (domain heuristic); the explored branch is written, the alternatives are parked on the choice points below it |

Between successive `solve_one_step` calls the queue is *not* refilled from scratch: `backtrack` schedules only
the propagators affected by the parked alternative, or by the objective bound it re-applies to the choice point it
resumes.

## Constants

`nucs/constants.py` holds what several layers share: the protocols a propagator (`PROP_*`, `EVENT_MASK_*`), a domain
heuristic (`DECISION_*`) and the solver (`DOMAIN_*`, `OBJECTIVE_*`) are written against, plus the logging levels. A
constant owned by one module lives with it instead: `CHOICE_POINT_*` in `nucs/solvers/choice_points.py`, `OFFSETS_*`
and `PROBLEM_*` in `nucs/problems/problem.py`, `SOLVER_*` in `nucs/solvers/backtrack_solver.py`, `OPTIM_*` in
`nucs/solvers/solver.py`, `STATS_*` in `nucs/statistics.py`.

The `SIGN_*` signatures — the fixed ABIs through which jitted callables are dispatched (see *Functions are values*
below) — live with the registry that compiles against them: `SIGN_COMPUTE_DOMAINS` and `SIGN_GET_TRIGGERS` in
`nucs/propagators/propagators.py`, `SIGN_CONSISTENCY_ALG` in `nucs/solvers/consistency_algorithms.py`,
`SIGN_VAR_HEURISTIC` and `SIGN_DOM_HEURISTIC` in `nucs/heuristics/heuristics.py`.

### Domain rows

A domain is a single `(min, max)` `int32` pair; a variable is its domain's index.

| index | constant | meaning |
|-------|----------|---------|
| 0 | `DOMAIN_MIN` | lower bound |
| 1 | `DOMAIN_MAX` | upper bound |

A variable is **bound** when its two columns hold the same value. (`DOMAIN_GROUND = 2` is not a domain column — it
is the third event bit, which happens to reuse the value `2`.)

### Events

When a propagator tightens a domain, the change is described by an event mask; `(variable, event)` is the key into the
triggers table. There are `EVENT_MASK_NB = 8` masks (bit combinations `0..7`).

| bit | value | constant | set when |
|-----|-------|----------|----------|
| 0 | 1 | `EVENT_MASK_MIN` | the domain's min increased |
| 1 | 2 | `EVENT_MASK_MAX` | the domain's max decreased |
| 2 | 4 | `EVENT_MASK_GROUND` | the domain became a singleton (`min == max`) |

### Result codes

Two distinct code sets share the values `0/1/2`:

| value | propagator (`compute_domains_*`) | consistency algorithm (`bc_algorithm`) |
|-------|----------------------------------|----------------------------------------|
| 0 | `PROP_INCONSISTENCY` — a domain wiped out | `PROBLEM_INCONSISTENT` — backtrack |
| 1 | `PROP_CONSISTENCY` — filtered, still active | `PROBLEM_UNBOUND` — branch |
| 2 | `PROP_ENTAILMENT` — satisfied for all remaining tuples, deactivate | `PROBLEM_BOUND` — solution |

### Statistics

`nucs/statistics.py` owns the array: `STATS_IDX_*` index a single `int64` array of `STATS_MAX = 10` counters
(`statistics`), `statistics_init` allocates it and `statistics_as_dictionary` reads it back under the `STATS_LBL_*`
labels. It is a leaf module — it takes the algorithm count and the algorithm names as arguments rather than importing
the propagator registry, so a jitted module can import a counter index without pulling the registry in behind it.

| idx | label | counts |
|-----|-------|--------|
| 0 | `ALG_BC_NB` | bound-consistency algorithm invocations |
| 1 | `PROPAGATOR_ENTAILMENT_NB` | propagator entailments |
| 2 | `PROPAGATOR_FILTER_NB` | propagator calls |
| 3 | `PROPAGATOR_FILTER_NO_CHANGE_NB` | propagator calls that changed nothing |
| 4 | `PROPAGATOR_INCONSISTENCY_NB` | propagator-detected inconsistencies |
| 5 | `SOLUTION_NB` | solutions found |
| 6 | `SOLVER_BACKTRACK_NB` | backtracks |
| 7 | `SOLVER_CHOICE_DEPTH` | current choice-point depth |
| 8 | `SOLVER_CHOICE_NB` | choices (branches) made |
| 9 | `SOLVER_ELAPSED_TIME` | solve time (accumulated in ns, reported in ms) |

## Important decisions

### Interval domains only — bound consistency, no holes

A domain is a single `(min, max)` `int32` pair; there are no sparse sets or bitmaps, and a propagator cannot remove a
value from the middle of a domain. The trade: weaker pruning than arc consistency, but domains form a flat contiguous
array that is O(1) to index and cheap to copy — which is what makes the choice-point and multiprocessing decisions
below work.

### Data-oriented state — all solver state lives in preallocated NumPy arrays

No Python objects in the hot path: everything is allocated once at solver init, and jitted functions take many
positional array arguments instead of a solver object — that is deliberate. Per-propagator metadata is laid out
CSR-style: an `offsets` array delimits, for each propagator, its slice of the flat `propagator_variables` and
`propagator_parameters` arrays; the trigger map is stored the same way.

| array | shape | indexed by | holds |
|-------|-------|-----------|-------|
| `algorithms` | `(P,)` uint8 | propagator | its `ALG_*` id |
| `priorities` | `(P,)` uint32 | propagator | its queue bucket index |
| `offsets` | `(P+1, 4)` uint32 | propagator | `[OFFSETS_VARIABLE, OFFSETS_PARAM, OFFSETS_STATE, OFFSETS_STATE_HINT]` — where each propagator's slice of the two arrays below starts; it ends where the next propagator's begins. The last two address its state block (see *Propagator state*) |
| `propagator_variables` | `(Σ arity,)` uint32 | flat (CSR) | every propagator's variables, concatenated |
| `propagator_parameters` | `(Σ params,)` int32 | flat (CSR) | every propagator's parameters, concatenated |
| `triggers` | `(Σ triggers,)` int32 | flat (CSR) | propagators to wake, grouped by `(variable, event)` |
| `triggers_offsets` | `(domain_nb · 8 + 1,)` int32 | `variable · 8 + event` | row offsets into `triggers` |

`P` = `propagator_nb`. A dense `(domain_nb, 8, propagator_nb)` trigger table would be mostly empty, so the propagators
watching `(variable, event)` are the slice `triggers[triggers_offsets[variable·8 + event] : … + 1]`.

### Backtrackable state is trailed, not copied

Every backtrackable value lives in one flat `int32` array, and one undo log restores all of it:

```
              0                     2n                2n+P              2n+P+S      2n+P+S+1
state (int32) [ ----- domains ----- | --- entailed --- | propagator state | unbound ]     n = domain_nb, P = propagator_nb, S = total propagator state width
```

`domains` is an `int32[:, ::1]` view of the head and `entailed` a view of the middle — the same memory, addressed the
way each reader wants it — so the flat index of `(variable, bound)` is `(variable << 1) | bound` and that of
propagator `p` is `2n + p`. A trail entry is `(flat index, old value)` with no discriminator, so restoring a domain
bound, reactivating an entailed propagator and rolling back the unbound-variable count are the same instruction.
Propagator state sits between the entailment flags and the unbound count, so `unbound_index()` — `len(state) - 1` —
stays independent of it. See *Propagator state* below.

| array | shape | dtype | role |
|-------|-------|-------|------|
| `state` | `(2·domain_nb + P + 1,)` | int32 | all the backtrackable state |
| `trail_log` | `(T, 2)` | int32 | the undo log |
| `trail_top` | `(1,)` | int32 | the trail size |
| `trail_indices` | `(len(state),)` | int32 | index of the last trail entry per cell, `-1` when none |
| `choice_point_stk` | `(H, 4)` | int32 | per choice point: `[CHOICE_POINT_TRAIL_MARK, CHOICE_POINT_VARIABLE, CHOICE_POINT_BOUND, CHOICE_POINT_VALUE]` |
| `choice_point_top` | `(1,)` | uint32 | the search depth |

A push copies nothing: it records the trail position and the single-bound tightening to apply on return, and writes
only the branch it explores. The alternatives are not materialised until the search reaches them.

**This trades time for the memory ceiling, deliberately.** `domains_stk` was `(H, domain_nb, 2)` int32 preallocated
before solving started — 64 KB per variable whatever the search did, so `bibd(10,15,6,4,2)` reserved 51.7 MB and a
10k-variable FlatZinc model would have wanted 655 MB. That is now 0.6 MB, and the ceiling on model size is gone. The
cost is 5-15% of throughput: measured on `golomb(9)`, 27 of the 72 domain cells are trailed per node, at 3 scattered
cells per push plus the undo, against one contiguous 72-cell memcpy — copying wins in that regime, and NuCS's models
are in it. `H` and `T` are starting sizes rather than ceilings; `solve_one_step` stops when either fills and the solver
doubles it, losing nothing of the search.

The write barrier lives in one place, `tighten`, which is the only site that writes a domain — propagation, branching,
the objective clamp, a custom consistency algorithm. Entailment is the exception: it has a semantic guard (a flag is
only written where it has just been read as clear) so it skips the positional one.

See `CHOICE_POINTS.md` for the mechanism in detail: the exact rule the barrier implements and why each part of it is
load-bearing, what the three decision kinds are, and what the two `OPTIM_*` modes do.

### Propagators are stateless pure functions on a scratch buffer

`compute_domains_*` receives a gathered copy of only its variables' domains (`domain_buffer`, sized once to the maximal
arity) and mutates that copy; `update_domains` diffs it against the real domains to derive events. A propagator that
detects inconsistency halfway through cannot corrupt global state, event computation is centralized in one place, and
there is no per-propagator state to restore on backtrack.

### Propagator state: a solver-owned block per propagator

`compute_domains_*` takes a third argument, `prop_state`: a per-propagator `int32[::1]` slice of `state`, addressed
CSR-style exactly like `propagator_variables`/`propagator_parameters` — an `OFFSETS_STATE` column in `offsets`,
holding *absolute* indices into `state` rather than 0-based ones (the base, `2 * domain_nb + propagator_nb`, is
already known to `Problem`, so `bc_algorithm` slices `state[offsets[p, OFFSETS_STATE]:offsets[p+1, OFFSETS_STATE]]`
directly, no extra argument for the base). A fourth column, `OFFSETS_STATE_HINT`, splits that block: it is where
p's untrailed hint suffix starts, so the trailed prefix the engine has to save is
`[offsets[p, OFFSETS_STATE], offsets[p, OFFSETS_STATE_HINT])` and both bounds come out of the offsets row the
slicing has already loaded — a per-propagator width belongs in the per-propagator table, and one kept beside it
would be a second cache line on the per-call path and another argument through `SIGN_CONSISTENCY_ALG`.
`register_propagator` takes an optional `get_state_fct` returning `(trailed_nb, hint_nb)`, defaulting to `(0, 0)`
— a propagator that doesn't ask for one costs nothing: no branch, no cache line, no trail entry.

Two tiers, by contract:

- **Tier A — hints (untrailed).** The filtering result must be identical whatever the block contains; staleness
  costs time, never correctness. Nothing is trailed, nothing is restored, the non-JIT path is untouched, and a fresh
  block's root value is whatever `np.zeros` gives it. `alldifferent`/`gcc` use this tier today (below).
- **Tier B — semantic state (trailed).** The cells are a function of the current domains, maintained incrementally;
  the engine trails the block, the propagator keeps it in sync. **Not yet used by any propagator** — the barrier for
  it exists (`bc_algorithm` trails a propagator's `trailed_nb`-wide prefix, unconditionally, through the same
  `trail_set` a domain write uses, right before calling `compute_domains_*`) but `choice_point_init` does not yet
  reseed a Tier-B block to a non-zero root value, so a first real Tier-B propagator needs that piece added.

Because the barrier runs before the call, a propagator that returns `PROP_INCONSISTENCY` halfway through may already
have written its state block, and that is safe — trailed on entry, restored by `trail_undo` like any other cell.

**`alldifferent`/`gcc` (Tier A — landed the two experiments below described as "explored, not adopted"):**
`get_state_alldifferent` reserves `[flag, min_sorted_vars[n], max_sorted_vars[n], bounds, t, d, h, ranks]` — the
`bounds/t/d/h/ranks` scratch that used to come from one `np.empty` per call, plus a warm-started sort permutation.
`flag == 0` means cold (a fresh, zeroed block): seed identity via `argsort_into` and set `flag = 1`; otherwise
`argsort_into_warm` re-sorts the existing (possibly stale) permutation in place, which is `O(n + inversions since
the previous call)` rather than relative to identity order — this is what removes the identity-seeded sort's
`O(n^2)` cliff when sort keys decorrelate from variable index. Inversions bound that cost but do not cap it, so
above `SORT_MAX_N` the sort runs on a budget of `SORT_WARM_BUDGET_FACTOR` shifts per variable and hands over to
`np.argsort` once it blows it. That is what makes the warm start pay at the arities it was meant for: a hard
`np.argsort` above `SORT_MAX_N` insures against the post-jump case on every call, including the overwhelming
majority that are one step down a descent. Measured against that hard fallback, a node that moved one bound
re-sorts 4× faster at `n=128`, 17× at `n=512` and 44× at `n=8192`, and a fully decorrelated permutation costs
+31%/+8%/+4% at those sizes — worst at small `n`, where the wasted shifts are largest next to a cheap
`np.argsort`. `get_state_gcc` reserves the equivalent scratch
(`bounds/t/d/h`, the sort permutations, `ranks`, `stable_intervals`, `stable_sets`, `new_mins`) behind the same
`flag`, plus the two `partial_sum` tables `l`/`u`: those are a function of `parameters` alone, which the engine
never writes, so they are built once on the cold call instead of by two `np.zeros` allocations per call. The
three arrays that used to come from a fresh `np.zeros` (`stable_intervals`, `stable_sets`, `new_mins`) are
explicitly re-zeroed each call, since a persistent block no longer implies that for free.

### Functions are values via numeric ids and wrapper addresses

Propagators and heuristics register into typed lists indexed by `ALG_*` / heuristic ids; the ids live in integer
arrays, and callables cross into nopython mode through `_get_wrapper_address` plus the `function_ptr_from_address`
intrinsic (see `nucs/numba_helper.py`). Numba cannot dispatch on heterogeneous Python callables, so indirection through
ids and addresses is the mechanism. Each callable family has a fixed `SIGN_*` signature, kept with its registry, that
every member must match — this is why an unused parameter can only be dropped from a family if *no* member needs it.

### The propagation queue is a bucketed FIFO keyed by complexity

`get_complexity_*` estimates a propagator's work per call; `compute_priority` folds that into a bucket index by
repeated right-shift of `BUCKET_FACTOR` bits (a log scale), clamped to `[0, BUCKET_NB)`. The queue (`nucs/buckets.py`)
runs the cheapest bucket first, FIFO within a bucket; add and pop are O(1), no heap. The whole queue is a single `int32`
array over `C = propagator_nb` elements, with intrusive per-element next-pointers and membership flags for set
semantics (`BUCKET_NB = 8`, so `STORAGE_OFFSET = 2 · BUCKET_NB = 16`):

| slice | length | holds |
|-------|--------|-------|
| `[0 : 8]` | `BUCKET_NB` | head element of each bucket (`-1` = empty) |
| `[8 : 16]` | `BUCKET_NB` | tail element of each bucket (`-1` = empty) |
| `[16 : 16+C]` | `C` | intrusive next-pointer per element (`-1` = end of bucket) |
| `[16+C : 16+2C]` | `C` | membership flag per element (`0`/`1`) |
| `[-1]` | 1 | cached lowest non-empty bucket index (search hint for `buckets_pop`) |

### The pure-Python escape hatch is a hard constraint

Everything must also run under `NUMBA_DISABLE_JIT=1` (debugging, coverage, real tracebacks) — this is why
`nucs/numba_helper.py` degrades typed lists to plain Python lists. Do not introduce Numba-only constructs without a
non-JIT fallback.

## Explored, not adopted

### A solver-owned scratch buffer for propagator working memory, and warm alldifferent permutations

*(benchmarked 2026-07, ~4% and ~8% respectively; landed together as the `prop_state` argument — see
*Propagator state* above)* These two were explored and shelved separately, purely because of the shared cost of
breaking `compute_domains_*`'s signature — a public extension point, so the change breaks every external propagator
and every test calling one directly. Once *something* forced that break, both were pure profit, so they landed in
the same pass as the mechanism itself. Numbers from the original measurement, kept for reference: scratch alone
(replacing `alldifferent`/`gcc`'s one `np.empty` per call) was a consistent 3.5–4.5% end-to-end speedup on queens
11–13 `solve_all`, matching a ~40 ns per-call microbenchmark saving (one Numba NRT allocation) — and the scratch
argument has to be typed C-contiguous (`int32[::1]`) in `SIGN_COMPUTE_DOMAINS`; reusing `parameters` as scratch
avoids the signature change but loses compile-time contiguity and was 5% *slower* despite allocating nothing. Warm
permutations on top brought queens 11–13 to ~7.5–8% end-to-end (roughly double scratch-only), langford(3,9) ~5.5%,
all_interval(12) 0% (alldifferent isn't its hot propagator); per call, warm equals cold when sort keys correlate with
variable index but removes the identity-seeded sort's O(n²) cliff when they don't (2.4× at n=128, 14× at n=512, 49×
at n=2048) — this is the gain FlatZinc-sourced models with large, arbitrarily-ordered alldifferents stand to see.
