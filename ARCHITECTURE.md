# NuCS architecture

NuCS is a **Constraint Satisfaction Problem solver**. It is written in pure Python and compiled at run time by
**Numba**, a just-in-time (JIT) compiler for numeric Python. All the state of the solver lives in flat NumPy arrays that
are allocated before the search starts. This is what lets the hot path (the code that runs at every node of the search)
run in Numba's *nopython* mode, where no Python object exists and the code runs as machine code.

This document gives the shape of the code and the decisions behind it. It is the place to start. The design documents
for contributors are in `design/`; the user documentation is in `docs/`.

## Repository structure

- **`nucs/problems/`** — a `Problem` has `domains` (one `(min, max)` pair per variable; a variable is *bound* when
  `min == max`) and a list of propagators. A propagator is added with
  `add_propagator(ALG_*, *variable_index_iterables, parameters=...)`. `Problem.init()` flattens all of this into the
  arrays that the solver reads (see *Data-oriented state* below).
- **`nucs/propagators/`** — one file per constraint, and `propagators.py`, which registers each constraint under a
  numeric `ALG_*` id. A propagator is the code that enforces one constraint: it reads the domains of its variables and
  removes the values that cannot be part of a solution. Each propagator is three required functions:
  `compute_domains_*` (the filtering; returns `PROP_INCONSISTENCY`, `PROP_CONSISTENCY` or `PROP_ENTAILMENT`),
  `get_triggers_*` (which changes of which variables must wake the propagator again) and `get_complexity_*` (the
  cost of a call, which orders the propagation queue). Two more functions are optional and have defaults in
  `register_propagator`: `get_state_*` (the size of the propagator's state block, see *Propagator state* below) and
  `is_vacuous_*` (whether the parameters and the initial domains make the constraint impossible to violate, in which
  case it is not posted). See `nucs/propagators/abs_eq_propagator.py` for the minimal template.
- **`nucs/solvers/`** — `BacktrackSolver`, which does the search (backtracking) and the propagation. The propagation
  to a fixpoint is `bc_algorithm` (`bc_algorithm.py`), registered as `CONSISTENCY_ALG_BC`. The search driver is
  `solve_one_step` (`search_loop.py`), the compiled loop that `BacktrackSolver` calls. The backtrackable state and
  its trail live in `state.py`; `tighten_at` is the only permitted way to write a domain. The stack of choice
  points, built on them, lives in `choice_points.py`. `allocate_search_arrays` (`search_arrays.py`) lays out the
  state and sizes the trail and the stack for a problem. `Interruption`
  (`interruption.py`) holds the one-cell array that stops the compiled search at its next node, and its two writers:
  `interrupt()` and the timer of a timeout. To iterate over the solutions, call `solver.solve()`. To optimize, call
  `solver.find_best(var, DOMAIN_MIN)` or `solver.find_best(var, DOMAIN_MAX)`.
- **`nucs/heuristics/`** — a variable heuristic picks the next unbound decision variable; a domain heuristic picks how
  to split its domain. Both are compiled by Numba against the fixed signatures `SIGN_VAR_HEURISTIC` and
  `SIGN_DOM_HEURISTIC` in `nucs/heuristics/heuristics.py`, and called by id.
- **`nucs/fzn/`** — the **FlatZinc adapter**. You write a model in MiniZinc and solve it with NuCS through
  `minizinc --solver nucs`. The pipeline is: `parser.py` (FlatZinc text → an intermediate representation) → `model.py`
  (`FznModel` builds a `Problem`) → `builtins.py` (the `BUILTINS` dispatch table, from a FlatZinc builtin name to
  `add_propagator` calls) → `search_annotations.py` (the search annotations and the search options → the searches,
  the restarts and last-conflict) → `runner.py` (the solve) → `output.py` (the stream of FlatZinc solutions).
  `fzn-nucs` is the console script that MiniZinc runs; `fzn-nucs --register` writes the solver configuration into `~/.minizinc/solvers`.
  `share/minizinc/nucs/` is the globals library: it keeps selected global constraints (alldifferent, gcc, lex, table)
  native, where MiniZinc would otherwise decompose them into many small constraints. To support one more builtin, add
  one entry to `BUILTINS` and, for a global that must stay native, one predicate file under `share/minizinc/nucs/`.
  See the `/add-propagator` skill, step 7.

### The solve loop

`solve_one_step` drives the search for one solution. It loops over two phases:

1. **Propagate to a fixpoint** (`bc_algorithm`). A *fixpoint* is the state where no propagator can remove a value
   any more. To reach it, the loop pops the cheapest triggered propagator from the queue, gathers the domains of its
   variables into `domain_buffer`, calls its `compute_domains_*`, writes the bounds it tightened back into the state,
   and puts into the queue the propagators that these changes trigger. It repeats this until the queue is empty or a
   domain becomes empty (a *wipe-out*).
2. **React to the status of the fixpoint**:

   | status | meaning | action |
   |--------|---------|--------|
   | `PROBLEM_BOUND` | the fixpoint is reached and all the variables are bound | emit the solution |
   | `PROBLEM_INCONSISTENT` | a domain wiped out | `backtrack`: pop a choice point, replay the undo log back to its mark, and schedule the propagators of the refuted decision. When optimizing, it keeps popping while the objective bound wipes out the choice point it resumes |
   | `PROBLEM_UNBOUND` | the fixpoint is reached but some variables are unbound | `branch`: the first search that still has an unbound decision variable picks one (variable heuristic) and says where to split its domain (domain heuristic). The branch that is explored is written into the state; the alternatives are parked on the choice points below it |

Between two calls of `solve_one_step` the queue is *not* filled again from the start. `backtrack` schedules only the
propagators that the parked alternative affects, or that the objective bound affects when it is applied again to the
choice point that the search resumes.

A **descent** is the search from one start at the root to the next restart. Without restarts, the whole search is one
descent. A descent is not one call of `solve_one_step`: a call also returns at each solution and each time an array
must grow, so one descent can take many calls.

## Constants

`nucs/constants.py` holds the constants that several layers share: the protocols that a propagator (`PROP_*`,
`EVENT_MASK_*`), a domain heuristic (`DECISION_*`) and the solver (`DOMAIN_*`, `OBJECTIVE_*`) are written against, and
the logging levels. A constant that belongs to one module lives in that module: `CHOICE_POINT_*` in
`nucs/solvers/choice_points.py`, `OFFSETS_*` and `PROBLEM_*` in `nucs/problems/problem.py`, `SOLVER_*` and
`SEARCH_CONTROL_*` in `nucs/solvers/search_loop.py`, `OPTIM_*` in `nucs/solvers/solver.py`,
`TIGHTENING_TRAIL_ENTRY_NB` in `nucs/solvers/state.py`, `STEP_TIGHTENING_NB` in `nucs/solvers/search_arrays.py`,
`STATS_*` in `nucs/statistics.py`.

The `SIGN_*` signatures are the fixed function types through which the compiled code calls a function by id (see
*Functions are values* below). Each lives with the registry that compiles against it: `SIGN_COMPUTE_DOMAINS` and
`SIGN_GET_TRIGGERS` in `nucs/propagators/propagators.py`, `SIGN_CONSISTENCY_ALG` in
`nucs/solvers/consistency_algorithms.py`, `SIGN_VAR_HEURISTIC` and `SIGN_DOM_HEURISTIC` in
`nucs/heuristics/heuristics.py`.

### Domain rows

A domain is one `(min, max)` pair of `int32`. A variable is the index of its domain.

| index | constant | meaning |
|-------|----------|---------|
| 0 | `DOMAIN_MIN` | lower bound |
| 1 | `DOMAIN_MAX` | upper bound |

A variable is **bound** when its two columns hold the same value. Note that `DOMAIN_GROUND = 2` is not a domain column:
it is the third event bit, which happens to have the value `2`.

### Events

When a propagator tightens a domain, an event mask describes the change. The pair `(variable, event)` is the key into
the triggers table. There are `EVENT_MASK_NB = 8` masks (the bit combinations `0..7`).

| bit | value | constant | set when |
|-----|-------|----------|----------|
| 0 | 1 | `EVENT_MASK_MIN` | the min of the domain increased |
| 1 | 2 | `EVENT_MASK_MAX` | the max of the domain decreased |
| 2 | 4 | `EVENT_MASK_GROUND` | the domain became one value (`min == max`) |

### Result codes

Two different sets of codes use the values `0/1/2`:

| value | propagator (`compute_domains_*`) | consistency algorithm (`bc_algorithm`) |
|-------|----------------------------------|----------------------------------------|
| 0 | `PROP_INCONSISTENCY` — a domain wiped out | `PROBLEM_INCONSISTENT` — backtrack |
| 1 | `PROP_CONSISTENCY` — filtered, still active | `PROBLEM_UNBOUND` — branch |
| 2 | `PROP_ENTAILMENT` — satisfied by all the remaining tuples, so the propagator is deactivated | `PROBLEM_BOUND` — solution |

*Entailment* means that the constraint holds for every assignment that the current domains still permit. An entailed
propagator has nothing left to do, so the solver stops waking it until a backtrack makes it active again.

### Statistics

`nucs/statistics.py` owns the statistics array. The `STATS_IDX_*` constants index one `int64` array of
`STATS_MAX = 11` counters (`statistics`); `statistics_init` allocates it and `statistics_as_dictionary` reads it back
under the `STATS_LBL_*` labels. The module is a leaf: it takes the number of algorithms and their names as arguments
instead of importing the propagator registry. Thus a compiled module can import a counter index without importing the
registry behind it.

| idx | label | counts |
|-----|-------|--------|
| 0 | `ALG_BC_NB` | calls of the bound-consistency algorithm |
| 1 | `PROPAGATOR_ENTAILMENT_NB` | propagator entailments |
| 2 | `PROPAGATOR_FILTER_NB` | propagator calls |
| 3 | `PROPAGATOR_FILTER_NO_CHANGE_NB` | propagator calls that changed nothing |
| 4 | `PROPAGATOR_INCONSISTENCY_NB` | inconsistencies found by a propagator |
| 5 | `SOLUTION_NB` | solutions found |
| 6 | `SOLVER_BACKTRACK_NB` | backtracks |
| 7 | `SOLVER_CHOICE_DEPTH` | the current depth of the stack of choice points |
| 8 | `SOLVER_CHOICE_NB` | choices (branches) made |
| 9 | `SOLVER_ELAPSED_TIME` | the solve time (accumulated in ns, reported in ms) |
| 10 | `SOLVER_RESTART_NB` | restarts from the root |

## Important decisions

### Interval domains only — bound consistency, no holes

A domain is one `(min, max)` pair of `int32`. There is no sparse set and no bitmap, so a propagator cannot remove a
value from the middle of a domain: it can only raise the min or lower the max. This is *bound consistency*. The trade
is this: the pruning is weaker than with arc consistency (which can remove any value), but the domains form one flat,
contiguous array, which is O(1) to index and cheap to copy. That is what makes the decisions below on choice points
and on multiprocessing work.

### Data-oriented state — all solver state lives in preallocated NumPy arrays

There is no Python object in the hot path. Everything is allocated once, when the solver is created, and the compiled
functions take many positional array arguments instead of a solver object. That is deliberate. The data of each
propagator is laid out in the CSR style (*compressed sparse row*, the layout of sparse matrices): one flat array holds
the data of all the propagators one after the other, and an `offsets` array says where the slice of each propagator
starts in it. `propagator_variables` and `propagator_parameters` are stored this way, and so is the trigger map.

| array | shape | indexed by | holds |
|-------|-------|-----------|-------|
| `algorithms` | `(P,)` uint8 | propagator | its `ALG_*` id |
| `priorities` | `(P,)` uint32 | propagator | the index of its bucket in the queue |
| `offsets` | `(P+1, 4)` uint32 | propagator | `[OFFSETS_VARIABLE, OFFSETS_PARAM, OFFSETS_STATE, OFFSETS_STATE_HINT]`: where the slice of each propagator starts in the two arrays below; it ends where the slice of the next propagator begins. The last two columns address its state block (see *Propagator state*) |
| `propagator_variables` | `(Σ arity,)` uint32 | flat (CSR) | the variables of every propagator, one after the other |
| `propagator_parameters` | `(Σ params,)` int32 | flat (CSR) | the parameters of every propagator, one after the other |
| `triggers` | `(Σ triggers,)` int32 | flat (CSR) | the propagators to wake, grouped by `(variable, event)` |
| `triggers_offsets` | `(domain_nb · 8 + 1,)` int32 | `variable · 8 + event` | where each group starts in `triggers` |
| `propagator_weights` | `(P + 2,)` float64 | propagator | the failure weight of each propagator, then the increment of the next failure and its growth (see *Failure weights*) |

`P` is `propagator_nb`. A dense trigger table of shape `(domain_nb, 8, propagator_nb)` would be almost empty, so the
propagators that watch `(variable, event)` are the slice `triggers[triggers_offsets[variable·8 + event] : … + 1]`.

### Failure weights — learned, global, never trailed

`bc_algorithm` knows which propagator failed, and records it in `propagator_weights` (`nucs/solvers/weights.py`). The
dom/wdeg variable heuristic reads these weights: it chooses the unbound variable with the smallest ratio of domain size
to *weighted degree*, which is the sum of the weights of its live propagators. A propagator is live for `x` when it is
not entailed and has an unbound variable other than `x`.

- **The weights are not backtracked.** A failure says that a constraint is hard, and that stays true in every part of
  the tree. So the weights are not in `state`, the trail never sees them, and `choice_point_init` does not clear them:
  an `OPTIM_RESET` restart keeps what the previous descents learned. Choco and Gecode do the same.
- **Decay costs one cell, not a pass over the weights.** Each propagator starts at 1, and a failure adds the current
  increment. With `weight_decay < 1` (the AFC of Gecode), each failure makes the increment larger by
  `1 / weight_decay`, which has the same effect as making all the older weights smaller. When the increment passes
  `1e100`, one pass divides all the cells by the same factor, which keeps every ratio. With the default decay of 1,
  the weight of a propagator is 1 plus its number of failures, which is the dom/wdeg of Choco.
- **The heuristic sees the network, and no variable heuristic pays for it.** `SIGN_VAR_HEURISTIC` gives every
  variable heuristic the entailment flags, the propagator tables, the trigger table and the weights.
- **The propagators of a variable come from the trigger table.** A propagator is listed under each
  `(variable, event mask)` that its trigger intersects. So the row of the mask of all the events (`watchers`) lists
  every propagator that watches the variable, each one once, in increasing order. A separate table from variable to
  propagators was built first and then removed: it held the same rows, plus the propagators that never wake on the
  variable (for example a linear constraint whose coefficient for that variable is 0). Such a propagator cannot fail
  because of the variable, so it must not add to its weight. Passing six more array arguments at each decision did not
  change the solve time of queens, golomb and magic sequence.
- **A failure that no propagator reports gets no weight.** When `backtrack` applies the objective bound again and a
  domain becomes empty, no propagator failed.

### Restarts and last-conflict — solver state, not choice-point state

Both live in `search_control`, a small `int64` array that `solve_one_step` reads and writes, and that the trail never
records. It holds: the failure limit of the current descent, the number of failures since the last start from the
root, whether last-conflict is on, the conflict variable, and the variable of the last decision.

- **A restart is the reset of `OPTIM_RESET`, triggered by a failure limit.** At the top of its loop, `solve_one_step`
  returns `SOLVER_RESTART` when the failures reach the limit. It does this before it starts a filtering, so the state
  it leaves is not a half-done one. `_restart` goes back to the root with `choice_point_init`, empties the queue and
  fills it again, and takes the next limit of the policy (`nucs/solvers/restarts.py`). The propagator weights stay.
- **The objective of the best solution is applied again at the root.** In `OPTIM_RESET`, the bound of the best
  solution is written once, at the root, and a restart undoes it. In `OPTIM_PRUNE`, the bound is armed, but
  `backtrack` only applies it to the choice points that it resumes. So `_reset_to_root` reads the armed bound in
  `objective` and applies it again after each restart, with a mark of 0. Without this, the search would find worse
  solutions again after each restart. `objective` is armed in both modes: in `OPTIM_RESET`, the root already holds the
  bound, so `backtrack` finds it applied and changes nothing.
- **An enumeration stops restarting at its first solution**, because a restart would find the solutions emitted so
  far a second time. Before the first solution, a restart loses nothing.
- **The conflict variable is the variable of the last *decision* that led to a failure**, not that of the choice
  point that the backtrack resumes. A branch records its variable; a failure moves it to the conflict cell; a
  refutation clears the record, so a failure that follows a refutation keeps the conflict variable. This is what makes
  last-conflict useful: after `x` fails on all its values, the search refutes an earlier decision on `y` and tries `x`
  again first, which tests at once whether `y` was the cause. Taking the variable of the resumed choice point instead
  replaces `x` by `y`, and measured no gain at all.
- **The conflict variable keeps the order of the searches.** `variable_searches` gives the search that owns each
  variable, and the conflict variable only takes the decision of that search. The searches before it have no unbound
  variable left when the loop reaches it, so the sequential order holds.

### Backtrackable state is trailed, not copied

A *trail* is an undo log: before a cell is written, its old value is recorded, so that a backtrack can restore it.
Every backtrackable value lives in one flat `int32` array, and one undo log restores all of it:

```
              0                     2n                2n+P              2n+P+S      2n+P+S+1
state (int32) [ ----- domains ----- | --- entailed --- | propagator state | unbound ]     n = domain_nb, P = propagator_nb, S = total propagator state width
```

`domains` is an `int32[:, ::1]` view of the head of `state`, and `entailed` a view of its middle: the same memory,
addressed the way each reader wants it. So the flat index of `(variable, bound)` is `(variable << 1) | bound`, and
that of propagator `p` is `2n + p`. A trail entry is `(flat index, old value)`, with nothing that says what kind of
cell it is. Thus restoring a domain bound, making an entailed propagator active again, and rolling back the count of
unbound variables are the same instruction. The propagator state sits between the entailment flags and the unbound
count, so `unbound_index()`, which is `len(state) - 1`, does not depend on it. See *Propagator state* below.

| array | shape | dtype | role |
|-------|-------|-------|------|
| `state` | `(2·domain_nb + P + 1,)` | int32 | all the backtrackable state |
| `trail_log` | `(T, 2)` | int32 | the undo log |
| `trail_top` | `(1,)` | int32 | the size of the trail |
| `trail_indices` | `(len(state),)` | int32 | the index of the last trail entry of each cell, `-1` when there is none |
| `choice_point_stk` | `(H, 4)` | int32 | per choice point: `[CHOICE_POINT_TRAIL_MARK, CHOICE_POINT_VARIABLE, CHOICE_POINT_BOUND, CHOICE_POINT_VALUE]` |
| `choice_point_top` | `(1,)` | uint32 | the depth of the search |

A choice point is where the search makes a decision and parks the alternatives, to come back to them later. A push
copies nothing: it records the position of the trail (the *mark*) and the single-bound tightening to apply on return,
and it writes only the branch that it explores. The alternatives are not built until the search reaches them.

**This trades time for the memory ceiling, deliberately.** Before, `domains_stk` was an `(H, domain_nb, 2)` int32
array, allocated before the solve started: 64 KB per variable, whatever the search did. So `bibd(10,15,6,4,2)` reserved
51.7 MB, and a FlatZinc model with 10k variables would have needed 655 MB. That is now 0.6 MB, and there is no ceiling
on the size of a model any more. The cost is 5-15% of the throughput. Measured on `golomb(9)`: 27 of the 72 domain
cells are trailed per node, at 3 scattered cells per push plus the undo, against one contiguous memcpy of 72 cells.
Copying wins in that regime, and the models of NuCS are in it. `H` and `T` are starting sizes, not ceilings:
`solve_one_step` stops when either array is full, and the solver doubles it and continues, losing nothing of the search.

The write barrier (the code that records the old value on the trail before a write) lives in one place, `tighten_at`,
which is the only site that writes a domain: the propagation, the branching, the objective clamp, and a custom
consistency algorithm. Entailment is the exception: it has a semantic guard (a flag is only written where it has just
been read as clear), so it does not need the positional one.

See `design/choice-points.md` for the mechanism in detail: the exact rule that the barrier implements and why each part
of it is necessary, what the three decision kinds are, and what the two `OPTIM_*` modes do.

### Propagators are stateless pure functions on a scratch buffer

`compute_domains_*` receives a copy of the domains of its own variables only (`domain_buffer`, sized once to the
largest arity), and changes that copy. `update_domains` then compares the copy with the real domains to derive the
events. The consequences: a propagator that finds an inconsistency halfway through cannot corrupt the global state,
the computation of the events is in one place, and there is no per-propagator state to restore on a backtrack.

### Propagator state: a solver-owned block per propagator

`compute_domains_*` takes a third argument, `prop_state`: a slice of `state`, of type `int32[::1]`, that belongs to
this propagator. It is addressed in the CSR style, exactly like `propagator_variables` and `propagator_parameters`:
the `OFFSETS_STATE` column of `offsets` holds, for each propagator, the start of its block. These are *absolute*
indices into `state`, not indices from 0: `Problem` already knows the base, `2 * domain_nb + propagator_nb`, so
`bc_algorithm` slices `state[offsets[p, OFFSETS_STATE]:offsets[p+1, OFFSETS_STATE]]` directly, with no extra argument
for the base. A fourth column, `OFFSETS_STATE_HINT`, splits the block in two: it is where the untrailed *hint* suffix
of `p` starts. So the trailed prefix that the engine must save is
`[offsets[p, OFFSETS_STATE], offsets[p, OFFSETS_STATE_HINT])`, and both bounds come out of the row of `offsets` that
the slicing has already loaded. A width that is per propagator belongs in the per-propagator table; a separate array
for it would be a second cache line on the per-call path and one more argument through `SIGN_CONSISTENCY_ALG`.
`register_propagator` takes an optional `get_state_fct`, which returns `(trailed_nb, hint_nb)` and defaults to
`(0, 0)`. A propagator that asks for no block costs nothing: no branch, no cache line, no trail entry.

There are two tiers, defined by their contract:

- **Tier A — hints (untrailed).** The result of the filtering must be the same whatever the block contains. A stale
  hint costs time, never correctness. Nothing is trailed, nothing is restored, the non-JIT path is untouched, and the
  root value of a fresh block is what `np.zeros` gives it. `alldifferent` and `gcc` use this tier today (see below).
- **Tier B — semantic state (trailed).** The cells are a function of the current domains, maintained incrementally:
  the engine trails the block, and the propagator keeps it in step with the domains. **Six propagators use it** (since
  2026-09-12): `lexleq` trails the two positions where its scan continues (8f2b14e); `relation` trails the size of its
  sparse set of live tuples (1bcdf98); `count_eq`, `count_eq_c`, `count_geq_c` and `count_leq_c` trail the two
  counters of their live set, at an arity of `LIVE_SET_MIN_ARITY` or more (6543f92). The engine gives this tier two
  things. First, `bc_algorithm` trails the `trailed_nb`-wide prefix of a propagator, unconditionally, through the same
  `trail_set` that a domain write uses, right before it calls `compute_domains_*`. Second, `choice_point_init` clears
  every trailed prefix, so a search that restarts from the root does not carry a block forward. That second thing is
  not housekeeping but soundness: `OPTIM_RESET` *drops* the trail instead of unwinding it, so nothing restores a block
  the way a backtrack would, and a propagator is entitled to read its block as a description of the node at which it
  is called. Left uncleared, a block gives wrong answers, not slow searches: `knapsack` returned 40 instead of 54 when
  this was tried with a Tier-B `linear_leq_c`. Zero is the root value by construction (it is what `np.zeros` gives the
  block, and what a propagator with state reads as "cold"), so clearing *is* the restore; a Tier-B block that needed a
  non-zero root value would still need seeding. The untrailed hint suffixes are deliberately left alone: a hint is
  valid from any node by contract, and clearing them would throw away exactly the warm sort permutations that an
  optimization restart most wants to keep. `TestPropagatorStateIsBacktracked` pins this on a Tier-B propagator that is
  local to the test, so the test does not depend on a shipped propagator.

Because the barrier runs before the call, a propagator that returns `PROP_INCONSISTENCY` halfway through may already
have written its state block, and that is safe: the block was trailed on entry, and `trail_undo` restores it like any
other cell.

#### Reporting what changed

A propagator call costs three passes over its variables, not one: `bc_algorithm` **gathers** them into
`domain_buffer`, the propagator **filters**, and `update_domains` **writes back**, walking the variables again to see
what moved and to wake the propagators that watch it. Measured in place, by running each pass twice, the write-back is
71% of the solve time of magic_sequence(200), against 10% for the gather. And most calls give it nothing to find: 96%
of the `sum_leq_c` calls of schur_lemma change no domain, 93.7% of the `count_eq` calls of magic_sequence, 85% of the
`sum_eq_c` calls of bibd, 31.6% of the `alldifferent` calls of queens.

So a propagator can say that it changed nothing. When it is registered with `reports_changes=True`, its
`get_state_fct` must reserve the **first cell of the hint suffix**. The engine sets that cell to `1` before every
call and reads it back after; a `0` lets it skip `update_domains` entirely. The equivalence is exact: the gather
copies `state` into `prop_domains`, so "the propagator wrote nothing" means "`prop_domains == state`", which means
"`update_domains` finds no event and schedules nobody".

- **The cell is untrailed, and could not be anything else.** It describes the call that has just happened, not the
  node, so there is nothing about it to restore. It never meets the trail, `choice_point_init` or the entry barrier.
- **Setting the cell to `1` first is what makes the default safe.** A propagator that forgets to answer gets the
  scan that it would have got anyway. Only the other direction is a bug, and a silent one: a propagator that reports
  `0` after it narrowed something drops that pruning, and the engine cannot tell, because the check would be the scan
  it avoids. For this reason `PropagatorTest` holds every reporting propagator to it, on every call of every curated
  case.
- **Opting in costs no change of the ABI.** It is a property of the algorithm, and `IDEMPOTENCIES` became
  `ALGORITHM_FLAGS`, which carries `PROP_FLAG_IDEMPOTENT | PROP_FLAG_REPORTS_CHANGES`, instead of a second array. A
  second array would have been a second parameter through `SIGN_CONSISTENCY_ALG`, and thus a breaking change for
  every custom consistency algorithm, for one bit. Six bits are left.

Twenty-one propagators report today: `linear_eq_c`/`leq_c`/`geq_c`/`neq_c`, `sum_eq`/`eq_c`/`leq_c`/`geq_c`,
`count_eq`/`leq_c`/`geq_c`, `alldifferent`, `gcc`, `lexleq`, `inverse`, `regular`, `relation`, and the four
`element_l_eq`/`_c`/`_alldifferent`/`_c_alldifferent`. All of them are n-ary. `abs_eq` and `leq_c` reported for a
while and were taken back out: a propagator with two variables has at most two write-back iterations to skip, and it
pays the report on every call to skip them. The set of reporting propagators is meant to stay small.

There are four ways to answer, and the shape of the propagator picks one:

- **Nothing at all.** In `count_geq_c` and `linear_neq_c`, every write is in a branch that returns entailment, so
  reaching `PROP_CONSISTENCY` already means that nothing was written, and they report unconditionally. Check for this
  shape before you write one of the three below.
- **A `changed` local**, set at each write and read at the one `PROP_CONSISTENCY` return. This is the linear and sum
  family, where the filtering is one flat loop.
- **A snapshot**, for the four `element_l_eq` variants. What matters is not how many write sites there are but how
  many domains can be written. `element_l_eq_alldifferent` has a dozen write sites, several inside its two scanning
  loops, but it can narrow only `i` and `v`, and it already keeps a snapshot of `v` for its own purposes, so the
  report costs two extra loads. Its one write to `l` is tested where it happens.
- **Clear on entry, set at each write**, for `lexleq` and `inverse`. The filtering of `lexleq` is spread over four
  mutually recursive functions; that of `inverse` over two helpers called four times, each returning a bare "still
  consistent" bool. In both cases a flag would have to be threaded back through every return. Clearing the cell at
  the entry point (which overrides the `1` the engine set) and setting it at each write gives the same answer, leaves
  the helper signatures alone, and survives any control flow. For `lexleq` the eight writes were all a `min` or a
  `max` onto a bound, so two inline helpers (`tighten_max`/`tighten_min`) narrow and report in one place.

**A cache keyed on the exact domains cannot hit, and the trigger mechanism is why.** `regular` was given one: it
recorded the domains at the end of a call that changed nothing, and compared the domains against them at the start of
the next call, to save its `O(length * q * s)` passes. Measured on the propagator alone it looked superb: a hit is
`O(length)`, so 4.4× at `length=16` and up to 47× at 1024, and 87–90% of its calls change nothing. In the engine it hit
**0 times out of 131**. The reason: a propagator is scheduled only when one of *its own* variables has changed, so
when it is entered again its domains are necessarily different from the ones it last settled on. The event that wakes
it is the event that invalidates the cache. Two attempts on `cumulative` and `disjunctive`, whose `O(n^3)` made the
microbenchmark show 18,000×, would have been worth exactly as much.

"Most calls change nothing" is therefore not the same claim as "the same input occurs again", and only the second
would make such a cache pay. What could work is a cache keyed on the *subset* of the domains that a conclusion depends
on, so that a change elsewhere does not invalidate it. But that needs the propagator to know its own dependencies,
which is a different and much larger design. `regular` keeps only the change report, which is the part that does pay.

`relation` keeps a **sparse set of live tuples**, and it is the one case where keeping state across calls *does* pay.
A tuple that no longer fits inside the domains can never fit again, so the set of tuples worth testing only shrinks
down a branch, and the scan costs the tuples still alive instead of the whole table. That is simple tabular reduction,
and what the table propagators of Gecode and Choco do. Only the *count* of ruled-out tuples is trailed, one cell; the
permutation is a hint, for the reason that a sparse set always permits: every swap stays inside the window that its
choice point handed down, so restoring the count restores the set.

Why this one pays where the same shape failed on `linear_*`: **skipping a tuple skips a loop over the columns, so the
indirection is amortised over the width of the table**, whereas skipping a variable in a linear constraint saved one
multiply-add and cost a full indirection. Measured on the propagator, by table size and by the share of the table
ruled out: 1024 tuples of 4 columns gives 1.56× at 50% dead, 4.05× at 92% and 6.23× at 99%; 1024 tuples of 8 columns,
7.32× at 99%. A small table gains little: `sports(8)` makes 568,602 calls against a handful of 3-column tuples and
measures flat. So the win is for the FlatZinc `table` constraints, which is where the long tables come from.

`count_eq` keeps a **live set of the x_i that are still undetermined**, which is the shape of `relation` applied to a
much narrower body. An `x_i` is undetermined while `a` is inside its domain and it is not fixed to `a`. It leaves that
state in two ways: by losing `a` (it can never equal `a`, so it leaves `count_max`) or by grounding on `a` (it must
equal `a`, so it joins `count_min`). Neither departure can be undone inside a branch, so the first pass costs the
elements that have not departed instead of all n. So do the other two passes, which only ever acted on an undetermined
`x_i` anyway, and whose case analysis collapses once every element scanned is known to have `a` strictly inside its
bounds. Two cells are trailed: the size of the live prefix, biased by one so that a zeroed block reads as cold, and
`count_min`. `count_max` is not stored at all, since `count_max = count_min + live_nb`; this matters because the
engine copies the trailed prefix on every call.

**`diffn` has the invariant but not the shrinkage**, which is the third way the rule can come out negative, and the
one that looks most like a win right up to the measurement. *(measured 2026-09-14)* Its pass is `O(n^2)` over the
pairs, and a pair is **definitely separated** once `x_i.max + dx_i <= x_j.min`, or one of the four symmetric forms.
That is monotone down a branch, because `x_i.max` only falls and `x_j.min` only rises. The work of the propagator on
such a pair is then provably nothing: the separation makes its own direction feasible, so the `not x_sep` / `not y_sep`
branches cannot be entered on that axis, and the tightenings that the other axis would apply are already implied by
the bounds. Checked on 242,323 randomly generated definitely-separated pairs: zero were narrowed.

None of this helps, because the set does not collapse. Dead pairs measured **38.5%** on `rect_09` and **29.7%** on
`square_21` for a first solution, and, going the wrong way as the search gets deeper, **26.7%** and **18.3%** when
enumerating all the solutions, the last over 9,863 nodes and 15,651 `diffn` calls. With 82% of the pairs still live,
the scan is hardly shorter than the one it replaces, and the four extra comparisons per live pair that maintain the
set eat what is left.

This case is worth separating from the rejection on `sum_*`, because the per-element term points the other way: a pair
costs a dozen loads and a branch, so the indirection is amortised the way it is for `relation`, and the break-even
shrinkage is *much* lower than for `count_eq`. `diffn` fails anyway, on the one term that was favourable everywhere
else. Worth knowing too: `square` is the only model here that posts `diffn`, and with its recommended searches it
makes 447 calls inside 14 ms, so even a real gain would have had nothing to show it on.

**The same live set transfers to `count_leq_c`, `count_geq_c` and `count_eq_c`, but only above an arity.** *(built
2026-09-14, shelved the same day for want of a model, landed once `count_shifts` existed)* The three count the same
predicate over the same monotone set, so the code is that of `count_eq` almost line for line, plus a short-circuit that
`count_eq` does not have: `count_min` is carried and can only rise, `count_max` is carried and can only fall, so either
bound can settle the constraint before anything is scanned. Measured on the propagator, against the plain scan, by
arity and by the fraction still undetermined:

| | n=16 | n=64 | n=256 | n=1024 |
|---|---|---|---|---|
| a tenth still live | 1.00–1.02× | 1.12–1.16× | 1.54–1.65× | 2.64–3.06× |
| a hundredth still live | 1.01–1.03× | 1.08–1.18× | 1.67× | 3.51–4.43× |

**What that table cannot see is the engine.** It calls `compute_domains_*` directly, so it leaves out the trailing of
the two new cells of the block, which the engine copies on every call. That cost is fixed while the saving grows with
n, so there is an arity below which it cannot pay. In the solver, a `count_leq_c` of arity 3 made
`employee_scheduling` 5–8% slower, the cost arriving undiluted. Hence `LIVE_SET_MIN_ARITY`: below it `get_state_*`
reserves no live set and `compute_domains_*` takes the plain scan, which is kept as an inlined helper rather than a
call, because at these arities the call itself measured against it (4.3% on `employee_scheduling`, down to 1.5% once
inlined).

Above that arity the win is real, and it took a model to see it: `count_shifts` in `datasets/fzn` posts `count_leq_c`
and `count_geq_c` over a roster of 120 variables, 2.9M and 2.1M calls at 87% and 96% no-change, and the pair is
**1.18×** there (2,112 ms to 1,748 ms), with `regular_shifts` flat as a control. Nothing in `nucs/examples` posts a
count wider than 21, which is why this stayed measured but undecided until the FlatZinc benchmark set existed. Every
statistic is identical across all seven FlatZinc models and all seven Python models.

**Recording the delta was then built, and lost.** *(measured 2026-09-14; the code is not kept, the paragraph below is
the build)* An earlier version of this note ruled it out on a half-read measurement. The version after that said that
the measurement ruled out only a *guard*, not a smaller input, and that `count_eq` on magic_sequence was the way to
find out. It was. `update_domains` gained a parallel `trigger_positions` array and appends, per trigger hit, the
position that the changed variable holds in the variable list of the woken propagator; `count_eq` keeps
`count_min`/`count_max` and a status code per variable, and updates them from the entries instead of rebuilding them.
Invalidation is an epoch stamp: a delta is usable only within the filtering that recorded it, which is exactly the
window over which domains only narrow, so nothing has to be trailed, and a backtrack invalidates everything for free.
Every statistic is identical on magic_sequence 100/200/400, `employee_scheduling` and queens 9 `solve_all`: the
filtering is the same one, computed a different way. It is **0.80× / 0.78× / 0.75×** on magic_sequence(200/400/600).

The decomposition is the useful part, because the target was real. Running the counting pass of `count_eq` twice in
place prices it at **34–38% of the whole solve**. A build where the engine still appends but the propagator ignores
what it appends (the cost with the benefit switched off) runs at 0.76×/0.74×/0.71×, so **the append alone costs
31–40%**, and the delta recovers only 5–6% of it. Two measurements say why:

- **61.2% of the `count_eq` calls rescan anyway**, because a propagator is called only **1.63 times per filtering**.
  The first call of each (propagator, filtering) pair has no delta by construction.
- **A usable delta still names a quarter of the variables**: 51 of 201 at `n=200`, 100 of 401 at `n=400`. The ratio
  does not depend on the scale, so it does not improve with size.

Under both is one structural fact: **the cost of recording a delta and the size of that delta are the same quantity.**
A change to one variable is appended once per propagator that watches it, and makes each of those deltas longer by
one. So recording costs `fan-out` and saves `arity − |delta|`, and the append has to be amortised over the calls that
the propagator makes before the filtering ends. At 1.63 calls there is nothing to amortise over. Predicting from
`|delta| / arity`, the 82-of-201 figure that the previous note reasoned from, was measuring the wrong thing a third
time: the usable deltas turned out *smaller* than that (51 of 201, a scan 4× narrower, better than the 2.4× predicted),
and it lost anyway, because calls-per-filtering, not scan width, is what binds.

And that closes a loop. **Change reporting, the mechanism that paid 3.0× on this very model, works by removing
calls**, and the calls it removed are the ones a delta needs to amortise against. The two optimisations compete for the
same slack, and the cheap one already took it.

Not all bad news: on models that post no delta propagator, the mechanism costs nothing measurable (queens 11
`solve_all` moved within the 1–4% drift band, in both directions across runs), so `PROP_FLAG_WANTS_DELTA` is free where
it is unused. The cost falls entirely on the models that opt in, and on this one it falls hardest precisely because
the fan-out of `count_eq` is total: every variable of magic_sequence is watched by every one of its 202 propagators,
so one bound change writes 202 entries.

Meanwhile **the mechanisms that have paid here need no delta at all**: reporting after the fact (the change cell of
`prop_state`), resuming past a prefix that is monotone by construction (`lexleq`), and dropping candidates that can
never come back (the live tuples of `relation`).

### The propagation loop is compiled without the reference-counting runtime

*(landed 2026-09-15)* `bc_algorithm` is `@njit(cache=True, _nrt=False)`. NRT is the runtime of Numba that counts the
references to each array. With the runtime on, an `inline="always"` helper increments the count of every array it
takes on entry and decrements it on exit: Numba inlines at the level of its own IR, so the argument handling of the
callee comes with it. On the propagation loop, those calls survived into the arm64 machine code as real calls
(`ldr x22, [_NRT_incref@GOTPAGEOFF]`, `blr x22`), which forced live values onto the stack around them. The IR of the
loop held 48 `NRT_incref` and 79 `NRT_decref` call sites: three of each for every bound that `trail_set` writes, one
typed-list incref and one `numba_list_size_address` C call per propagator call, and more around the slices given to
the propagator and around the queue operations. Removing all of them is worth **1.23–1.36×** on golomb, magic_square,
all_interval and magic_sequence, and **1.07×** on queens, with every statistic identical. See *The engine's per-call
cost has no single owner* below, which this result overturns.

Two rules come with it, and both are enforced:

- **Nothing in the loop may allocate an array or index a typed list.** Either fails to compile with "NRT required but
  not enabled". That is why a propagator is called through `call_compute_domains` and an `int64` array of compiled
  addresses, not through a typed list of functions.
- **Every compiled function that the loop calls must be `inline="always"`.** A function compiled on its own is
  inlined by LLVM as whichever context first compiled it into the cache left it, and the cache key does not record
  that context. `buckets_add`, first compiled by a caller with reference counting, brought 4 increfs and 6 decrefs
  back into the loop on a warm cache, and none on an empty one. That fails nothing; it only makes the speed of the
  loop depend on the history of the cache. `tests/solvers/test_bc_algorithm.py` forces the bad case (it compiles the
  queue helpers in a reference-counted context first) and fails if the compiled loop holds any refcount call.

`_nrt` is a private option of Numba. Numba is pinned to an exact version, and the same test pins the option on the
decorator, so an upgrade that drops or ignores it fails a test, not a benchmark. The propagators keep the runtime:
`_nrt=False` on the five hottest ones that do not allocate measured only 1.02–1.03× more.

### Functions are values

The functions that the solver calls by id (the propagators, the heuristics, the consistency algorithms) are *values*,
not names: each registry compiles its functions against one fixed signature (`SIGN_*`), and the compiled search calls
them through a function pointer, or, for the propagators, through their compiled address (see the rules above). A
new propagator or heuristic is therefore one more entry in a table, and the compiled search does not change.

### The pure-Python escape hatch is a hard constraint

Everything must also run under `NUMBA_DISABLE_JIT=1`, which runs the code as plain Python (for debugging, for
coverage, and for real tracebacks). This is why `nucs/numba_helper.py` degrades the typed lists to plain Python lists.
Do not introduce a construct that only Numba can run without a non-JIT fallback.

## Explored, not adopted

This section records the ideas that were measured and rejected, with the numbers. Each of them looks good on paper;
the notes say why it does not pay in NuCS, so that nobody tries it a second time without a new reason.

### Four ways of making a propagator incremental that do not work

*(measured 2026-09-12, the fourth 2026-09-14)* Kept because each looks compelling on paper, and two of them look
spectacular in a microbenchmark.

**A cache keyed on the exact domains cannot hit.** A propagator is woken only when one of its own variables has
changed, so the event that wakes it is the event that invalidates the cache: 0 hits in 131 lookups on `regular`,
against a microbenchmark that said 47×. This is a consequence of the absence of a per-modification hook; see
*Reporting what changed* above for what that does and does not rule out.

**Ground-task elimination does not apply to `cumulative` or `disjunctive`.** In a linear constraint, the contribution
of a ground variable is a *scalar*, so it folds into a running constant, and the variable leaves the computation. The
contribution of a ground task is a *rectangle in time*, and what the other tasks need to know is where it sits, not
what it totals: a ground task is the most constraining kind there is. Two tasks are enough to show it: capacity 1, A
ground at 0 with `p=2,h=1`, B free in `[0,5]`. The earliest start of B goes from 0 to 2, and that pruning comes only
from A. The companion idea, caching the profile, fails because `_filter_est` derives its segment boundaries from *all*
the compulsory parts, so one task that is not ground re-segments the profile.

**Recording what changed and handing it to the propagator costs more than the scan it saves.** Built for `count_eq`
on magic_sequence, the model with the widest propagators and the highest no-change rate, so the best case there is,
and measured at 0.75–0.80×, with identical statistics throughout. The append costs 31–40% of the solve, and the
shorter scan gives back 5–6%. See *Reporting what changed* above for the decomposition, and for why the binding
constraint is calls-per-filtering (1.63) rather than the width of the delta.

**A Θ-Λ-tree does not beat the cubic enumeration of `disjunctive` at the sizes NuCS sees.** The `O(n log n)`
edge-finding was written and verified (26,000 random instances, identical status and identical earliest starts to the
enumeration it replaces), and then measured slower almost everywhere:

| instance shape | n=16 | n=64 | n=256 | n=512 |
|---|---|---|---|---|
| one shared deadline, tight (the cubic case) | 0.21× | 0.27× | 0.92× | **1.62×** |
| tight windows, spread starts (the ordinary case) | 0.24× | 0.12× | 0.06× | **0.05×** |

`_filter_est` is cubic only when every task shares a deadline. On spread instances, its `lct[i] <= bound` test keeps
`Θ` tiny for most bounds, so it runs almost linearly (4 µs at 512 tasks), while the tree pays seven array allocations
per call and six memory writes per level up the path on every move. The crossover exists, but only at 512 tasks in
the worst shape, and the shape cannot be known cheaply. The lesson is the same as for the linear-compaction and
array-merge results: **a tight contiguous scan with a data-dependent early exit is very hard to beat with a better
asymptotic bound and worse locality.**

### The engine's per-call cost has no single owner

*(measured 2026-09-15)* The section below prices the body of `sum_eq` at under 12% of `golomb(11)`, which leaves
**about 110 ns of every 121 ns call** in the machinery around the propagator. Two candidates looked like separate,
removable items, and both were tested against a stop rule set in advance: **if together they came to under 10%, the
overhead is diffuse and not worth a specialised engine.**

**Devirtualising the dispatch: 2.5%.** `compute_domains_fcts[algorithm](...)` is a call through a Numba typed list
of functions. It cannot be inlined, and it forces live values to memory across the boundary. It looked like the
largest single item, because real models dispatch over almost nothing: `golomb(11)` posts 101 propagators over
**3 distinct algorithms**, `queens(12)` posts 3 over **1**. A chain that tests those three before the indirect call,
and falls through to it otherwise, measured:

| | devirtualised | base | |
|---|---|---|---|
| golomb(10), all three algorithms in the chain | 158.8 ms | 163.3 ms | 1.029× |
| golomb(11), all three | 4,617 ms | 4,733 ms | 1.025× |
| magic_square(4), two of three | 203.6 ms | 206.9 ms | 1.017× |
| magic_sequence(400), **none** — the control | 218.8 ms | 220.1 ms | 1.006× |

The gradient is right (the gain falls as fewer of the algorithms of a model are devirtualised, and the control that
takes none of the branches is flat), so the effect is real and not drift. It is also a quarter of what the hypothesis
needed. The dispatch of Numba is an indirect branch through a known signature, not a Python lookup, and an indirect
branch that resolves to one of three targets is predicted well. What devirtualising removes is mostly the barrier to
inlining, and these bodies are too small for inlining to be worth much.

**Removing the per-call statistics: 1.4%, and that is an upper bound.** Four `int64` read-modify-writes per call,
37M times on `golomb(11)`. The build that drops them also drops the `no_change` bookkeeping, of which they were the
only consumer, so it removes more than the counters:

| | no statistics | base | | calls/ms |
|---|---|---|---|---|
| golomb(10) | 161.8 ms | 164.0 ms | 1.014× | 8,073 |
| golomb(11) | 4,699 ms | 4,740 ms | 1.009× | 7,880 |
| magic_square(4) | 206.6 ms | 207.5 ms | 1.004× | 4,367 |
| magic_sequence(400) | 221.2 ms | 220.0 ms | 0.995× | 2,056 |

There is no model that a change to the statistics cannot affect, so the substitute for a control was a prediction
made before the run: the gain should follow the number of calls per millisecond. It does, and it reaches zero at the
bottom of the column, which is what makes 1.4% an effect and not noise.

**So the stop rule fires.** Together the two account for about one twenty-fifth of the 110 ns. What is left is not
overhead in the sense of being removable: three scattered reads out of `state` and six stores into the domain buffer,
five `offsets` loads, a queue pop, three `tighten_at` calls with their write barriers, and a walk of a trigger slice
(**mean 4.8 entries on golomb, 3.0 on queens**) that tests membership and entailment on each entry. That is what
propagating a constraint costs.

This reframes the finding instead of solving it. "90% of a `sum_eq` call is machinery" reads as a verdict against the
engine; it is closer to saying that **a three-variable constraint is not enough work to amortise a general propagation
engine over**. The forty-five ternary `sum_eq` of golomb are a modelling choice, and the lever on models of that
shape is to post fewer and wider constraints, not to make the call cheaper. The one per-call change that ever did pay,
inlining `update_domains` for 5.8%, is already in.

**It had one owner after all.** *(measured 2026-09-15, the same day)* The stop rule priced the two items that it
could name, the dispatch and the statistics, and read the remaining 110 ns off the source: reads, stores, a queue pop,
a trigger walk. The machine code held one more item that the source does not show at all: reference counting, added
by the compiler around every array that an inlined helper takes. Compiling the loop without the runtime of Numba took
`golomb(11)` from 4,692 to 3,561 ms with the search identical, about **30 ns of the 121 ns** per call; see *The
propagation loop is compiled without the reference-counting runtime* above. The lesson is where to look: an
itemisation read off the source cannot find a cost that the compiler adds, and `inspect_llvm` on a fresh compilation
(a dispatcher loaded from the cache has no IR) shows what the loop really executes.

### The busiest propagator in the benchmarks has nothing in it worth optimising

*(measured 2026-09-15)* `sum_eq` makes **31.5 million calls** across the Python benchmark set, more than any other
propagator, and 26.7 million of those are `golomb(11)` alone: **72% of every propagator call it makes**, in the
longest-running model there is. It looks like the obvious place to spend an afternoon.

It is not, and the reason is the arity. Golomb posts `sum_eq` as `d(0,i) + d(i,j) = d(0,j)`, forty-five of them at
**arity 3**, so the two loops of the propagator run two iterations each. Pricing them by running each twice in place,
with each probe checked neutral by an unchanged node count (which the equivalent probe on `regular` failed to be):

| pass | golomb(10) | golomb(11) |
|---|---|---|
| accumulating the bounds | 3.2% | 2.8% |
| filtering the variables | 8.3% | 5.9% |
| **the whole body** | **11.5%** | **8.7%** |

So making `sum_eq` *free* would buy about a tenth of the model it dominates. `golomb(11)` spends 4,530 ms on 37.3
million propagator calls, **121 ns a call**, of which roughly 110 ns is everything around the propagator: gathering
three domains out of `state`, the indirect call through the typed list of functions, and `update_domains` walking
three variables back with their trigger slices. **A propagator this small is a rounding error inside its own call.**

This moves the question instead of answering it. The cost is the per-call machinery, and the one measurement there
says the same thing from the other side: inlining `update_domains` into the filtering loop was worth 5.8%, while every
attempt to shrink or merge the arrays it touches has measured zero, because they are already in the L1 cache. A model
that posts a hundred propagators of arity 2 and 3 pays that machinery 37 million times, and that, not the arithmetic
inside any of them, is what golomb is made of.

### `bin_packing_load` was rebuilding a subset-sum per item, and a stamp made it worse before prefixes made it 4.6x

*(measured 2026-09-15)* Its item rule asks, for each candidate in a bin, whether the *other* candidates can still fill
the remaining load of the bin. It answered by rebuilding the whole subset-sum reachability without that candidate.
That is `O(nc^2 * total)` per bin, and on `datasets/fzn/bin_packing_load` it came to **1.67 billion DP iterations**
across 1,327,522 `_reach` calls: 98.5 per propagator call.

**The first attempt made it 38% slower, and is worth recording.** The obvious move was the one that had just worked
for `regular`: put the reachability buffer in the state block and *stamp* it instead of clearing it, which removes
1.3M allocations and 221M cell clears. It measured 1,847 ms → 2,418 ms, with identical filtering. Two controls place
the blame exactly: the same scratch buffer with explicit clearing measured 1,843 ms, so **the allocations were worth
nothing**, and switching from `uint8` to `int32` cost nothing either. So what the 38% bought was the stamped
comparison itself. A 0/1 array tested against zero is a shape that LLVM can vectorise; `reach[s - w] == stamp` against
a value that varies in the loop is not. The clearing was only 13.2% of the work it came with, so there was never much
there to win.

**What did work was removing the recomputation.** Reachability without candidate `t` is the subsets of the candidates
before `t` combined with those after it. So one pass from the right records every suffix, a running prefix covers the
left, and each question becomes "does some reachable prefix sum `a` leave `[lo - a, hi - a]` reachable in the
suffix": `O(1)` per `a` against the running count of the suffix, so `O(nc * total)` for the whole rule instead of
`O(nc^2 * total)`. Measured **1,847 ms → 399 ms, 4.6x**, with `nvalue_assign` flat as a control and every statistic
identical across both benchmark sets.

**Two lessons, and they pull in opposite directions.** Removing an allocation pays in proportion to the number of
allocations per unit of work: `nvalue` allocated twice per call against a couple of hundred operations and gained 2x;
this propagator allocated a hundred times per call against 124,000 DP iterations and gained nothing. And a mechanism
that won next door can lose on its own merits: the stamp is right where clearing dominates, and wrong where clearing
is a thirteenth of the work and the array it marks is the working set of the hot loop.

### `regular` allocated twice a call, cleared what it allocated, and then recomputed its own answer

*(measured 2026-09-15)* The layered graph of Pesant needs two reachability tables: which states are reachable at each
position, and from which states acceptance is still reachable. `regular` built both with `np.zeros` on every call, on
a propagator whose state block held nothing but the report cell. This is the same omission as for `nvalue`, found the
same way: the FlatZinc coverage report showed 544,220 calls to a propagator that nobody had looked at.

Two changes, and the second is the interesting one:

- **The tables moved into the state block and are stamped instead of cleared.** Each call takes the next stamp and
  writes it where it used to write a 1; a cell that holds any other stamp reads as unreachable. That removes the
  allocation *and* the `O(length * q)` clearing, which a fresh `np.zeros` had provided for free. Worth about 1.09×
  on `regular_shifts`.
- **The pruning pass was deleted, because the backward pass already knows the answer.** A symbol `v` is supported at
  position `i` exactly when some forward-reachable state reads it into a state that still accepts, which is the pair
  `(q, v)` that the backward sweep already visits. The sweep now records the supported range as it goes, and the
  pruning is one bound assignment per variable instead of a scan that calls a support test per candidate bound. The
  early exit survives for the states that are *not* forward-reachable: they cannot support anything, so they still
  stop at the first transition that marks them.

Measured on `regular_shifts` (442,654 nodes, 544,220 `regular` calls), with `nvalue_assign` flat as a control and
every statistic identical across both benchmark sets: **901 ms → 776 ms, 1.16×**.

**The estimate that pointed here was badly wrong in size and right in direction.** Running each pass twice in place
priced the forward sweep at 7% of the model, the backward sweep at 14% and the pruning loop at **64%**, which
predicted roughly 2×. Deleting the pruning loop was worth 8%. Duplication prices a pass by running it twice, and for a
pass built out of calls to a small function that is not inlined, that doubles the call overhead too, which is not
what removing the pass recovers. Inlining that function first, which measured 3%, was the signal that the estimate
was inflated, and it was read too late.

### `nvalue` was left out of the scratch-and-warm-permutations pass, and paid 2x for it

*(measured 2026-09-14)* `nvalue` bounds its count variable between the largest set of pairwise-disjoint domains and
the size of their union, and each bound needs the variables sorted: by upper bound for the first, by lower bound for
the second. It did that with **two `np.argsort` calls per call**, which is two Numba allocations and two sorts seeded
from scratch, on a propagator with no state block at all. That is exactly what `alldifferent` and `gcc` stopped doing
when `prop_state` landed, and `nvalue` was simply not in that pass. It now keeps both permutations in its block and
re-sorts them from their own previous contents through the same `argsort_into_warm`, so a call costs the inversions
since the last one instead of a full sort.

Measured on `nvalue_assign` (30 variables over 10 values, 806,212 nodes, 725,713 `nvalue` calls), with
`regular_shifts` flat as a control and the search identical counter for counter:

| | time | |
|---|---|---|
| before | 751 ms | |
| warm permutations, no allocation | 417 ms | **1.80×** |
| and change reporting on top | **374 ms** | **2.01×** total |

Entailment was added in the same pass and is worth about nothing here: `low == up` proves that every remaining
assignment has the same number of distinct values, and it is sound and monotone, but it fired on 7 calls out of
725,713. Worth keeping for the models where the count collapses early, not worth claiming.

**The general point is that when a mechanism lands, it does not land everywhere.** Scratch and warm permutations were
measured at 3.5–4.5% and about 8% when they went into `alldifferent` and `gcc`; the same change is worth 1.80× on
`nvalue`, because `nvalue` sorts *twice* per call and its models call it often. Nothing recorded which propagators
had been converted, so the one with the most to gain kept allocating for a year. The FlatZinc coverage report exists
to make that kind of omission visible.

### A solver-owned scratch buffer for propagator working memory, and warm alldifferent permutations

*(benchmarked 2026-07, about 4% and about 8% respectively; landed together as the `prop_state` argument, see
*Propagator state* above)* These two were explored and shelved separately, only because of the shared cost of breaking
the signature of `compute_domains_*`: a public extension point, so the change breaks every external propagator and
every test that calls one directly. Once *something* forced that break, both were pure profit, so they landed in the
same pass as the mechanism itself. The numbers of the original measurement, kept for reference: scratch alone
(replacing the one `np.empty` per call of `alldifferent`/`gcc`) was a consistent 3.5–4.5% end-to-end speedup on
queens 11–13 `solve_all`, which matches a saving of about 40 ns per call in a microbenchmark (one Numba NRT
allocation). The scratch argument has to be typed C-contiguous (`int32[::1]`) in `SIGN_COMPUTE_DOMAINS`; reusing
`parameters` as scratch avoids the change of signature but loses the contiguity at compile time, and was 5% *slower*
although it allocated nothing. Warm permutations on top brought queens 11–13 to about 7.5–8% end-to-end (roughly
double scratch-only), langford(3,9) about 5.5%, all_interval(12) 0% (alldifferent is not its hot propagator). Per
call, a warm sort equals a cold one when the sort keys correlate with the variable index, but it removes the O(n²)
cliff of the sort seeded with the identity when they do not (2.4× at n=128, 14× at n=512, 49× at n=2048). This is the
gain that FlatZinc-sourced models with large alldifferents in arbitrary order stand to see.
