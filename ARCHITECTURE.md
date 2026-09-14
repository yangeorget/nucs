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

`lexleq` keeps two trailed positions besides the report — `q` and `r`, which are α and β in Carlsson and
Beldiceanu's *Revisiting the Lexicographic Ordering Constraint* (`papers/lexleq/`), the report this
implementation is transcribed from. That paper's whole point is resumption: it claims `O(n)` on posting plus
**amortized `O(1)` per propagation event**, and §5 says to record "the state q ∈ {1,2,3,4} that preceded the
suspension, and the positions α β γ", trailed. NuCS had transcribed the four states and threaded all three
positions through their signatures, with the resumption switched off — every call restarted the automaton at
state 1, position 0, and the `if r > i + 1: i = r` that jumps to a carried position could never fire.

Two of the three positions are safe to carry here, and the third is not, which is worth separating:

- **`q` (α) and `r` (β) are safe**, and for the same reason: state 1 advances `q` only over positions where
  its two tightenings have forced `x_i = y_i` *ground on both sides*, and state 2 advances `r` only over
  positions where all four bounds are equal. Both prefixes are therefore monotone within a branch, so
  resuming past them is not merely sound but silent — the skipped loop tests a condition that still holds
  and applies tightenings that write nothing. The paper puts the second as "in state 2, any letter before
  pos. β is ignored; this is safe, for the ignored letters will all be =".
- **`s` (γ) is not**, and the reason is a difference in engines rather than in the algorithm. The paper skips
  past γ in states 3 and 4 too, on the grounds that a position before it which has since become decisive
  will arrive as its own pending propagation event — "the pending event will lead to just that, when it is
  processed". NuCS coalesces events: a propagator is woken once however many of its variables moved, and
  sees all of it on entry, so there is no later event to rely on and skipping those positions would lose the
  transition. Their conditions (`x_i.max == y_i.min`, `x_i.min == y_i.max`) are not monotone under narrowing
  either, which is the same fact from the other side. **A paper's incrementality can depend on how its host
  engine delivers events, and that assumption has to be checked rather than inherited.**

Measured on the propagator, each resume against the rescan it replaces, with a 99% prefix: state 1's is 1.6×
at `n=128`, 3.2× at 512, 9.4× at 2048; state 2's is 1.6× at 128, 3.1× at 512, 9.4× at 2048 — the resumed
call flat at ~230 ns whatever the prefix. Its four mutually recursive states are the Frisch et al. lexicographic
algorithm, whose whole point is to resume where it left off, and NuCS had transcribed it with the resumption
switched off — every call restarted the automaton at state 1, index 0. State 1's loop tightens both bounds onto
one value, so a counted index is *ground on both sides*, which makes the prefix monotone within a branch and
`q` trailed rather than a hint. Resuming there is not merely sound but silent: the loop over an already-equal
prefix tests a condition that still holds and applies two tightenings that are no-ops. Measured on the
propagator, the resumed call is flat at ~220 ns whatever the prefix, against a rescan that grows with it —
1.6× at `n=128`, 3.2× at `n=512`, 9.4× at `n=2048`, each with a 99% prefix. The `r` and `s` pointers of the
other three states still start at 0 every call: their conditions are *not* monotone under narrowing, so
resuming past them could skip an index that has since become decisive.

Two things are worth copying from how `alldifferent` does it. Its block gained a cell, because
it was already using its first for the cold flag — the report cell is fixed at the front of the hint suffix so the
engine can find it without knowing anything about the propagator's own layout. And its `filter_lower`/`filter_upper`
now return `(consistent, changed)`, with the Hall-interval writes *tested* rather than made blind: the write is
frequently a no-op, and counting "I executed a write" instead of "I changed a value" is safe but throws away most
of the win.

Measured, median of five: magic_sequence(200) 48 → 16 ms, magic_sequence(100) 6 → 2 ms, quasigroup(5,12)
756 → 608 ms, quasigroup(5,11) 93 → 76 ms, golfers(3,2,5) 7 → 5 ms.

**Nothing below about 5% is measurable here, so nothing below it is claimed.** This machine drifts 1–4% with run
order: an A/B that runs one build then the other reports whichever ran second as slower, in *both* directions.
The way to catch that is a model the change cannot possibly affect — for the reporting work, `queens` posts
nothing but `alldifferent` — and to discard the whole comparison when the control moves as much as the subjects.
Numbers stated here are the ones far enough above that floor to survive the reversed order; the smaller readings
that were once quoted for `abs_eq`, `queens` and `langford` did not, and are gone.

**The saving is `(no-change rate) × (unbound variables in scope) × (cost of a `tighten_at`)`.** Arity is only a
proxy for the middle term, and a good one only while domains stay wide: `update_domains` skips `tighten_at`
outright for a *bound* variable, at the cost of a load and a compare, so a propagator whose variables ground
early has almost no write-back to skip however wide it is. A `count_leq_c` at arity 60 with a 76.2% no-change
rate — the profile that made `count_eq` pay — measured *nothing*, because in that model the whole write-back was
2.4% of the run: the search ground its variables fast. `magic_sequence` is the opposite regime, 200-wide domains
and 198 backtracks, which is why its write-back was 71%. Read the rate together with how long the model's domains
stay live, not with arity alone.
`count_eq` on magic_sequence has arity 101 over domains that stay unbound, so the scan *was* the work. Against
that, `sum_leq_c` on schur_lemma changes nothing on 96% of its calls and `abs_eq` on all_interval on 39.6% of
144,439 — but both hold two or three variables, so there is next to nothing to skip whatever the rate, which is
why `abs_eq` no longer reports. `alldifferent` on queens sits between, with the rate but only arity 12 against an
`O(n log n)` body that still runs in full. `gcc` and `lexleq` have the shape that
pays — arity `n` and `2n`, no-change rates of 48–100% — and no bundled model that exercises them: the ones that
post them run in 1–8 ms with at most 1778 calls.

The clearest case of the rule paying is `quasigroup`, whose `element_l_eq_alldifferent` makes 66.8% of all
propagator calls at arity 14 and **changes nothing on 90.9% of them**; `inverse` adds 6.9% at arity 24 and 69.8%.
Duplicating the write-back in situ put it at 172 ms of quasigroup(5,12)'s 756 ms, and reporting from those two
took the model to 608 ms — 1.24×, with 1.22× on (5,11), 1.20× on (5,10) and 1.16× on (3,8). That is slightly more
than the 123 ms the scan-share arithmetic predicted, because skipping `update_domains` drops its call overhead
and its per-variable scheduling branch as well as the scan.

**Test the write, don't make it blind.** `alldifferent`'s Hall-interval writes, `gcc`'s four, and
`trim_domains_inverse`'s clamp of every variable to `[offset, offset + n - 1]` are all no-ops on most calls.
Counting "I executed a write" rather than "I changed a value" is safe but gives most of the reporting back, and
it is the standard way to get this mechanism wrong.

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

**Every solver coalesces the call; NuCS alone discards the delta.** A propagator is enqueued once however many
of its variables moved — and so it is in Gecode (`me_combine` folds the modification events together), in Choco
("the variable stores events but is enqueued only once") and in CP-SAT ("assume that a propagator does not need
to be called twice in a row"). Coalescing the *call* is universal. What the other three keep, and NuCS throws
away, is **what changed**:

| | call coalesced | delta survives to the call |
|---|---|---|
| Gecode | yes | `ModEventDelta`, plus **advisors** run per modification with a `Delta` |
| Choco | yes | `propagate(idxVarInProp, mask)` per modified variable, when `reactToFineEvent()` |
| CP-SAT | yes | `IncrementalPropagate(watch_indices)` — the accumulated changed set |
| **NuCS** | yes | **no** — `update_domains` holds `(variable, events)` when it schedules, and records only that it did |

Two things follow, and they pull in opposite directions.

**What is genuinely closed off** is anything that needs a *per-modification* hook. Only Gecode's advisors have
one, and two attempts foundered on that:

- **A cache keyed on the exact domains never hits.** A propagator is woken *because* one of its variables
  changed, so its domains always differ from the ones it last settled on: the event that wakes it is the event
  that invalidates the cache. Measured 0 hits in 131 lookups on `regular`.
- **A propagator cannot defer work to "a later event".** Carlsson and Beldiceanu's `lexleq` skips positions
  before γ on the grounds that one which has become decisive "will lead to just that, when it is processed" —
  true with per-variable events, false here, where the batched call is the only call. So `q` and `r` are
  carried and `s` is not.

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
