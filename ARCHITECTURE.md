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
  the engine trails the block, the propagator keeps it in sync. **Not yet used by any propagator**, but the two
  pieces it needs are both in place: `bc_algorithm` trails a propagator's `trailed_nb`-wide prefix, unconditionally,
  through the same `trail_set` a domain write uses, right before calling `compute_domains_*`; and
  `choice_point_init` clears every trailed prefix, so a search restarted from the root does not carry a block
  forward. That second one is not housekeeping but soundness: `OPTIM_RESET` *drops* the trail rather than unwinding
  it, so nothing restores a block the way a backtrack would, and a propagator is entitled to read its block as
  describing the node it is being called at. Left uncleared it produces wrong answers, not slow searches —
  `knapsack` returned 40 instead of 54 when this was tried with a Tier-B `linear_leq_c`. Zero is the root value by
  construction (it is what `np.zeros` gives the block and what a state-keeping propagator reads as cold), so
  clearing *is* the restore; a Tier-B block wanting a non-zero root value would still need seeding. The untrailed
  hint suffixes are deliberately left alone: a hint is valid from any node by contract, and clearing them would
  throw away exactly the warm sort permutations an optimization restart most wants to keep.
  `TestPropagatorStateIsBacktracked` pins this on a test-local Tier-B propagator, so it holds whether or not a
  shipped propagator uses the tier.

Because the barrier runs before the call, a propagator that returns `PROP_INCONSISTENCY` halfway through may already
have written its state block, and that is safe — trailed on entry, restored by `trail_undo` like any other cell.

#### Reporting what changed

A propagator call costs three passes over its variables, not one: `bc_algorithm` **gathers** them into
`domain_buffer`, the propagator **filters**, and `update_domains` **writes back** — walking them again to see what
moved and wake whoever watches it. Measured in situ by duplicating each pass, the write-back is 71% of
magic_sequence(200)'s solve time against 10% for the gather. And most calls give it nothing to find: 96% of
schur_lemma's `sum_leq_c` calls change no domain, 93.7% of magic_sequence's `count_eq`, 85% of bibd's `sum_eq_c`,
31.6% of queens' `alldifferent`.

So a propagator can say so. Declaring `reports_changes=True` at registration obliges its `get_state_fct` to reserve
the **first cell of the hint suffix**; the engine pre-sets that cell to `1` before every call and reads it back
after, and a `0` lets it skip `update_domains` outright. The equivalence is exact: the gather copies `state` into
`prop_domains`, so "the propagator wrote nothing" is "`prop_domains == state`", which is "`update_domains` finds no
event and schedules nobody".

- **The cell is untrailed, and could not be anything else.** It describes the call that has just happened, not the
  node, so there is nothing about it to restore; it never meets the trail, `choice_point_init` or the entry barrier.
- **Pre-setting to `1` is what makes the default safe.** A propagator that forgets to answer gets the scan it would
  have got anyway. Only the other direction is a bug, and it is a silent one — reporting `0` after narrowing
  something drops that pruning and the engine cannot tell, because checking would be the scan it is avoiding.
  `PropagatorTest` therefore holds every reporting propagator to it, on every call of every curated case.
- **Opting in costs no ABI.** It is a per-algorithm property, and `IDEMPOTENCIES` became `ALGORITHM_FLAGS` carrying
  `PROP_FLAG_IDEMPOTENT | PROP_FLAG_REPORTS_CHANGES` rather than growing a second array — which would have been a
  second parameter through `SIGN_CONSISTENCY_ALG`, and so a breaking change to every custom consistency algorithm,
  for one bit. Six bits are left.

Twenty-two propagators report today: `linear_eq_c`/`leq_c`/`geq_c`/`neq_c`, `sum_eq`/`eq_c`/`leq_c`/`geq_c`,
`count_eq`/`leq_c`/`geq_c`, `alldifferent`, `gcc`, `lexleq`, `inverse`, `regular`, `scc`, `relation`, and the four
`element_l_eq`/`_c`/`_alldifferent`/`_c_alldifferent`. All of them are n-ary: `abs_eq` and `leq_c` reported for a
while and were taken back out, because a propagator holding two variables has at most two write-back iterations
to skip and pays the report on every call to do it. The reporting set is meant to stay tight.

Four ways of answering, picked by shape:

- **Nothing at all.** `scc` writes no domain *ever* — it is a feasibility check, answering only whether the
  digraph is still strongly connected — so it reports unconditionally and never pays a write-back scan again.
  `count_geq_c` and `linear_neq_c` are the weaker version of the same thing: every write they make is in a
  branch that returns entailment, so reaching `PROP_CONSISTENCY` already means nothing was written. Worth
  checking for before writing any of the three below; `scc` and `dummy` are the only propagators that never
  write at all, and `dummy` holds no variables.

- **A `changed` local**, raised at each write and read at the single `PROP_CONSISTENCY` return — the linear and
  sum family, where the filtering is one flat loop.
- **A snapshot**, for the four `element_l_eq` variants: what matters is not how many write sites there are but
  how many domains can be written. `element_l_eq_alldifferent` has a dozen sites, several inside its two scanning
  loops, and can narrow only `i` and `v`
  — and already snapshots `v` for its own purposes, so the report costs two extra loads. Its one write to `l` is
  tested where it happens.
- **Clear on entry, raise at each write**, for `lexleq` and `inverse`. `lexleq`'s filtering is spread over four
  mutually recursive functions, `inverse`'s over two helpers called four times, each returning a bare "still
  consistent" bool; either way a flag would have to be threaded back through every return. Clearing the cell at
  the entry point — overriding the `1` the engine pre-set — and raising it at each write is the same answer, leaves
  the helper signatures alone, and survives any control flow. For `lexleq` the eight writes were all a `min` or a
  `max` onto a bound, so two inline helpers (`tighten_max`/`tighten_min`) narrow and report in one place.

**A cache keyed on the exact domains cannot hit, and the trigger mechanism is why.** `regular` was given
one: it recorded the domains a call that changed nothing had ended on, and compared against them on the way
in, to spare its `O(length * q * s)` passes. On the propagator that looked superb — a hit is `O(length)`, so
4.4× at `length=16` up to 47× at 1024, and 87–90% of its calls change nothing. In the engine it hit **0 times
out of 131**. A propagator is scheduled only when one of *its own* variables has changed, so by the time it
is entered again its domains necessarily differ from the ones it last settled on: the event that wakes it is
the event that invalidates the cache. Two tries at `cumulative` and `disjunctive`, whose `O(n^3)` made the
microbenchmark show 18,000×, would have been worth exactly as much.

"Most calls change nothing" is therefore not the same claim as "the same input recurs", and only the second
would make such a cache pay. What could work is a cache keyed on the *subset* a conclusion depends on, so
that a change elsewhere does not invalidate it — but that needs the propagator to know its own dependencies,
which is a different and much larger design. `regular` keeps only the change report, which is the part that
does pay.

`relation` keeps a **live-tuple sparse set**, and it is the one case where absorbing *does* pay. A tuple
that no longer fits inside the domains can never fit again, so the set of tuples worth testing shrinks
monotonically down a branch; the scan costs the tuples still alive rather than the whole table. That is
simple tabular reduction, and what Gecode's and Choco's table propagators do. Only the *count* of ruled-out
tuples is trailed — one cell — while the permutation is a hint, for the reason the sparse set always allows:
every swap stays inside the window its choice point handed down, so restoring the count restores the set.

Why this one pays where the same shape failed on `linear_*`: **skipping a tuple skips a loop over the
columns, so the indirection is amortised over the width of the table**, whereas skipping a variable in a
linear constraint saved a single multiply-add and cost a full indirection. Measured on the propagator, by
table size and share of the table ruled out: 1024 tuples of 4 columns gives 1.56× at 50% dead, 4.05× at 92%
and 6.23× at 99%; 1024 of 8 columns, 7.32× at 99%. A small table gains little — `sports(8)` makes 568,602
calls against a handful of 3-column tuples and measures flat — so the win is for FlatZinc `table`
constraints, which is where the long ones come from.

`count_eq` keeps a **live set of the x_i still undetermined**, which is `relation`'s shape applied to a
much narrower body. An `x_i` is undetermined while `a` is inside its domain and it is not fixed to `a`; it
leaves that state by losing `a` (it can never equal `a`, so it leaves `count_max`) or by grounding on `a` (it
must, so it joins `count_min`). Neither departure can be undone inside a branch, so the first pass costs the
ones that have not departed instead of all n — and so do the other two passes, which only ever acted on an
undetermined `x_i` anyway, and whose case analysis collapses once every element scanned is known to have `a`
strictly inside its bounds. Two cells are trailed: the size of the live prefix, biased by one so that a
zeroed block reads as cold, and `count_min`. `count_max` is not stored at all — `count_max = count_min +
live_nb` — which matters because the engine copies the trailed prefix on every call.

**`diffn` has the invariant and not the shrinkage**, which is the third way the rule can come out negative
and the one that looks most like a win right up to the measurement. *(measured 2026-09-14)* Its pass is
`O(n^2)` over pairs, and a pair is **definitely separated** once `x_i.max + dx_i <= x_j.min` — or any of the
four symmetric forms — which is monotone down a branch because `x_i.max` only falls and `x_j.min` only
rises. The propagator's work on such a pair is then provably nothing: the separation makes its own direction
feasible, so the `not x_sep` / `not y_sep` branches cannot be entered on that axis, and the tightenings the
other axis would apply are already implied by the bounds. Checked on 242,323 randomly generated
definitely-separated pairs: zero were narrowed.

None of which helps, because the set does not collapse. Dead pairs measured **38.5%** on `rect_09` and
**29.7%** on `square_21` for a first solution, and — going the wrong way as the search deepens — **26.7%**
and **18.3%** enumerating all solutions, the last over 9,863 nodes and 15,651 `diffn` calls. At 82% of the
pairs still live, the scan is barely shorter than the one it replaces, and the four extra comparisons per
live pair that maintain the set eat what is left.

This one is worth separating from the `sum_*` rejection, because the per-element term points the other way:
a pair costs a dozen loads and a branch, so the indirection amortises the way `relation`'s does and the
break-even shrinkage is *much* lower than `count_eq`'s. `diffn` fails anyway, on the one term that was
favourable everywhere else. Worth knowing too that `square` is the only model here that posts `diffn`, and
with its recommended searches it makes 447 calls inside 14 ms — so even a real gain would have had nothing
to show it on.

**The same live set transfers to `count_leq_c`, `count_geq_c` and `count_eq_c`, but only above an arity.**
*(built 2026-09-14, shelved the same day for want of a model, landed once `count_shifts` existed)* The three
count the same predicate over the same monotone set, so the code is `count_eq`'s almost line for line, plus
a short-circuit it does not have: with `count_min` carried and only able to rise, and `count_max` carried
and only able to fall, either bound can settle the constraint before anything is scanned. Measured on the
propagator, against the plain scan, by arity and by the fraction still undetermined:

| | n=16 | n=64 | n=256 | n=1024 |
|---|---|---|---|---|
| a tenth still live | 1.00–1.02× | 1.12–1.16× | 1.54–1.65× | 2.64–3.06× |
| a hundredth still live | 1.01–1.03× | 1.08–1.18× | 1.67× | 3.51–4.43× |

**What that table cannot see is the engine.** It calls `compute_domains_*` directly, so it leaves out the
trailing of the block's two new cells, which the engine copies on every call. That cost is fixed while the
saving grows with n, so there is an arity below which it cannot pay — and in the solver a `count_leq_c` of
arity 3 made `employee_scheduling` 5–8% slower, the cost arriving undiluted. Hence `LIVE_SET_MIN_ARITY`:
below it `get_state_*` reserves no live set and `compute_domains_*` takes the plain scan, which is kept as
an inlined helper rather than a call, because at these arities the call itself measured against it
(4.3% on `employee_scheduling`, down to 1.5% once inlined).

Above it the win is real, and it took a model to see: `count_shifts` in `datasets/fzn` posts `count_leq_c`
and `count_geq_c` over a 120-long roster, 2.9M and 2.1M calls at 87% and 96% no-change, and the pair is
**1.18×** there (2,112 ms to 1,748 ms) with `regular_shifts` flat as a control. Nothing in `nucs/examples`
posts a count wider than 21, which is why this sat measured-but-undecided until the FlatZinc benchmark set
existed. Every statistic is identical across all seven FlatZinc models and all seven Python models.

**Recording the delta was then built, and lost.** *(measured 2026-09-14; the code is not kept -- the
paragraph below is the build)* An earlier version of this note ruled it out on a half-read measurement, and the
version after that said the measurement ruled out only a *guard*, not a smaller input, and that `count_eq` on
magic_sequence was the way to find out. It was: `update_domains` gained a parallel `trigger_positions` array
and appends, per trigger hit, the position the changed variable holds in the woken propagator's own variable
list; `count_eq` keeps `count_min`/`count_max` and a status code per variable and updates them from the
entries instead of rebuilding them. Invalidation is an epoch stamp — a delta is usable only within the
filtering that recorded it, which is exactly the window over which domains only narrow, so nothing has to be
trailed and backtracking invalidates the lot for free. Every statistic is identical on magic_sequence
100/200/400, `employee_scheduling` and queens 9 `solve_all`: the filtering is the same one, computed a
different way. It is **0.80× / 0.78× / 0.75×** on magic_sequence(200/400/600).

The decomposition is the useful part, because the target was real. Duplicating `count_eq`'s counting pass in
situ prices it at **34–38% of the whole solve**. A build where the engine still appends but the propagator
ignores what it appends — cost with the benefit switched off — runs at 0.76×/0.74×/0.71×, so **the append
alone costs 31–40%** and the delta recovers only 5–6% of it. Two measurements say why:

- **61.2% of `count_eq` calls rescan anyway**, because a propagator is called only **1.63 times per
  filtering**. The first call of each (propagator, filtering) pair has no delta by construction.
- **A usable delta still names a quarter of the variables** — 51 of 201 at `n=200`, 100 of 401 at `n=400`.
  The ratio is scale-invariant, so it does not improve with size.

Underneath both is one structural fact: **the cost of recording a delta and the size of that delta are the
same quantity.** A change to one variable is appended once per propagator watching it, and grows each of those
propagators' deltas by one. So recording costs `fan-out` and saves `arity − |delta|`, and the append has to be
amortised over the calls the propagator makes before the filtering ends. At 1.63 calls there is nothing to
amortise over. Predicting from `|delta| / arity` — the 82-of-201 figure the previous note reasoned from — was
measuring the wrong thing a third time: the usable deltas turned out *smaller* than that (51 of 201, a 4×
narrower scan, better than the 2.4× predicted), and it lost anyway, because calls-per-filtering, not
scan-width, is what binds.

And that closes a loop. **Change reporting — the mechanism that paid 3.0× on this very model — works by
removing calls**, and the calls it removed are the ones a delta needs to amortise against. The two
optimisations compete for the same slack, and the cheap one already took it.

Not all bad news: on models posting no delta propagator the mechanism costs nothing measurable (queens 11
`solve_all` moved within the 1–4% drift band, in both directions across runs), so `PROP_FLAG_WANTS_DELTA` is
free where it is unused. The cost falls entirely on the models that opt in, and on this one it falls hardest
precisely because `count_eq`'s fan-out is total: every variable of magic_sequence is watched by every one of
its 202 propagators, so one bound change writes 202 entries.

Meanwhile **the mechanisms that have paid here need no delta at all** — reporting after the fact
(`prop_state`'s change cell), resuming past a prefix that is monotone by construction (`lexleq`), and dropping
candidates that can never come back (`relation`'s live tuples).

### The pure-Python escape hatch is a hard constraint

Everything must also run under `NUMBA_DISABLE_JIT=1` (debugging, coverage, real tracebacks) — this is why
`nucs/numba_helper.py` degrades typed lists to plain Python lists. Do not introduce Numba-only constructs without a
non-JIT fallback.

## Explored, not adopted

### Four ways of making a propagator incremental that do not work

*(measured 2026-09-12, the fourth 2026-09-14)* Kept because each looks compelling on paper and two of them
look spectacular in a microbenchmark.

**A cache keyed on the exact domains cannot hit.** A propagator is woken only when one of its own variables
has changed, so the event that wakes it is the event that invalidates the cache — 0 hits in 131 lookups on
`regular`, against a microbenchmark that said 47×. A consequence of there being no per-modification hook; see
*The propagation queue* above for what that does and does not rule out.

**Ground-task elimination does not apply to `cumulative` or `disjunctive`.** In a linear constraint a ground
variable's contribution is a *scalar*, so it folds into a running constant and the variable leaves the
computation. A ground task's contribution is a *rectangle in time*, and what the other tasks need to know is
where it sits, not what it totals — a ground task is the most constraining kind there is. Two tasks suffice
to show it: capacity 1, A ground at 0 with `p=2,h=1`, B free in `[0,5]` — B's earliest start goes 0 → 2, and
that pruning comes only from A. The companion idea, caching the profile, fails because `_filter_est` derives
its segment boundaries from *all* the compulsory parts, so one non-ground task re-segments the profile.

**Recording what changed and handing it to the propagator costs more than the scan it saves.** Built for
`count_eq` on magic_sequence — the model with the widest propagators and the highest no-change rate, so the
best case there is — and measured 0.75–0.80×, with statistics identical throughout. The append costs 31–40%
of the solve and the shorter scan gives back 5–6%. See *The propagation queue* above for the decomposition and
for why the binding constraint is calls-per-filtering (1.63) rather than the width of the delta.

**A Θ-Λ-tree does not beat `disjunctive`'s cubic enumeration at the sizes NuCS sees.** The `O(n log n)`
edge-finding was written and verified — 26,000 random instances, identical status and identical earliest
starts to the enumeration it replaces — and then measured slower almost everywhere:

| instance shape | n=16 | n=64 | n=256 | n=512 |
|---|---|---|---|---|
| one shared deadline, tight (the cubic case) | 0.21× | 0.27× | 0.92× | **1.62×** |
| tight windows, spread starts (the ordinary case) | 0.24× | 0.12× | 0.06× | **0.05×** |

`_filter_est` is cubic only when every task shares a deadline. On spread instances its `lct[i] <= bound` test
leaves `Θ` tiny for most bounds, so it runs near-linearly — 4 µs at 512 tasks — while the tree pays seven
array allocations per call and six memory writes per level up the path on every move. The crossover exists,
but at 512 tasks in the worst shape only, and the shape is not knowable cheaply. Same lesson as the
linear-compaction and array-merge results: **a tight contiguous scan with a data-dependent early exit is very
hard to beat with a better asymptotic and worse locality.**

### `bin_packing_load` was rebuilding a subset-sum per item, and a stamp made it worse before prefixes made it 4.6x

*(measured 2026-09-15)* Its item rule asks, for each candidate in a bin, whether the *other* candidates can
still fill the bin's remaining load — and answered it by rebuilding the whole subset-sum reachability
without that candidate. That is `O(nc^2 * total)` per bin, and on `datasets/fzn/bin_packing_load` it came to
**1.67 billion DP iterations** across 1,327,522 `_reach` calls: 98.5 per propagator call.

**The first attempt made it 38% slower, and is worth recording.** The obvious move was the one that had just
worked for `regular`: put the reachability buffer in the state block and *stamp* it rather than clear it,
removing 1.3M allocations and 221M cell-clears. It measured 1,847 ms → 2,418 ms, filtering identical. Two
controls place the blame exactly: the same scratch buffer with explicit clearing measured 1,843 ms, i.e.
**the allocations were worth nothing**, and switching from `uint8` to `int32` cost nothing either — so what
the 38% bought was the stamped comparison itself. A 0/1 array tested against zero is a shape LLVM can
vectorise; `reach[s - w] == stamp` against a loop-varying value is not. The clearing was only 13.2% of the
work it accompanied, so there was never much there to win.

**What did work was removing the recomputation.** Reachability without candidate `t` is the subsets of the
candidates before `t` combined with those after it, so one pass from the right records every suffix, a
running prefix covers the left, and each question becomes "does some reachable prefix sum `a` leave
`[lo - a, hi - a]` reachable in the suffix" — `O(1)` per `a` against the suffix's running count, so `O(nc *
total)` for the whole rule instead of `O(nc^2 * total)`. Measured **1,847 ms → 399 ms, 4.6x**, with
`nvalue_assign` flat as a control and every statistic identical across both benchmark sets.

**Two lessons, and they pull against each other.** Removing an allocation pays in proportion to allocations
per unit of work: `nvalue` allocated twice per call against a couple of hundred operations and gained 2x;
this propagator allocated a hundred times per call against 124,000 DP iterations and gained nothing. And a
mechanism that won next door can lose on its own merits — the stamp is right where clearing dominates and
wrong where it is a thirteenth of the work and the array it marks is the hot loop's working set.

### `regular` allocated twice a call, cleared what it allocated, and then recomputed its own answer

*(measured 2026-09-15)* Pesant's layered graph needs two reachability tables — which states are reachable
at each position, and from which states acceptance is still reachable. `regular` was building both with
`np.zeros` on every call, on a propagator whose state block held nothing but the report cell: the same
omission as `nvalue`, found the same way, by the FlatZinc coverage report showing 544,220 calls against a
propagator nobody had looked at.

Two changes, and the second is the interesting one:

- **The tables moved into the state block and are stamped rather than cleared.** Each call takes the next
  stamp and writes it where it used to write a 1; a cell holding any other stamp reads as unreachable. That
  removes the allocation *and* the `O(length * q)` clearing, which a fresh `np.zeros` had been providing
  for free. Worth ~1.09× on `regular_shifts`.
- **The pruning pass was deleted, because the backward pass already knows the answer.** A symbol `v` is
  supported at position `i` exactly when some forward-reachable state reads it into a state that still
  accepts — which is the pair `(q, v)` the backward sweep is already visiting. It now records the supported
  range as it goes, and pruning is a bound assignment per variable instead of a scan calling a support test
  per candidate bound. The early exit survives for states that are *not* forward-reachable, which cannot
  support anything and so still stop at the first transition that marks them.

Measured on `regular_shifts` (442,654 nodes, 544,220 `regular` calls) with `nvalue_assign` flat as a
control and every statistic identical across both benchmark sets: **901 ms → 776 ms, 1.16×**.

**The estimate that pointed here was badly wrong in size and right in direction.** Duplicating each pass in
situ priced the forward sweep at 7% of the model, the backward at 14% and the pruning loop at **64%** —
which predicted roughly 2×. Deleting the pruning loop outright was worth 8%. Duplication prices a pass by
running it twice, and for a pass built out of calls to a small non-inlined function that doubles the call
overhead too, which is not what removing the pass recovers. Inlining that function first, which measured
3%, was the signal that the estimate was inflated and it was read too late.

### `nvalue` was left out of the scratch-and-warm-permutations pass, and paid 2x for it

*(measured 2026-09-14)* `nvalue` bounds its count variable between the largest set of pairwise-disjoint
domains and the size of their union, and each bound needs the variables sorted — by upper bound for the
first, by lower bound for the second. It was doing that with **two `np.argsort` calls per call**, which is
two Numba allocations and two sorts seeded from scratch, on a propagator with no state block at all. That
is exactly what `alldifferent` and `gcc` stopped doing when `prop_state` landed, and `nvalue` simply was
not in that pass. It now keeps both permutations in its block and re-sorts them from their own previous
contents through the same `argsort_into_warm`, so a call costs the inversions since the last one rather
than a full sort.

Measured on `nvalue_assign` (30 variables over 10 values, 806,212 nodes, 725,713 `nvalue` calls), with
`regular_shifts` flat as a control and the search identical counter for counter:

| | time | |
|---|---|---|
| before | 751 ms | |
| warm permutations, no allocation | 417 ms | **1.80×** |
| and change reporting on top | **374 ms** | **2.01×** total |

Entailment was added in the same pass and is worth ~nothing here — `low == up` proves every remaining
assignment has the same number of distinct values, and it is sound and monotone, but it fired on 7 calls
out of 725,713. Worth keeping for the models where the count collapses early, not worth claiming.

**The general point is that a mechanism landing does not land it everywhere.** Scratch and warm
permutations were measured at 3.5–4.5% and ~8% when they went into `alldifferent` and `gcc`; the same
change is worth 1.80× on `nvalue`, because `nvalue` sorts *twice* per call and its models call it often.
Nothing recorded which propagators had been converted, so the one with the most to gain kept allocating
for a year. The FlatZinc coverage report exists to make that kind of omission visible.

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
