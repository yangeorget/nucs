# Lazy clause generation in NuCS — design

> **Status:** proposed, 2026-10-05. The implementation started on 2026-11-03: Phase 0 and steps 3 and 4 of 7.0 are done.

This document is the design of lazy clause generation (LCG) for NuCS. It is written for a reader who knows
classical constraint programming (propagation, search, Gecode and Choco internals) but not LCG or SAT solving.
Part 1 explains the concepts, with a worked example and links to read. Parts 2 to 6 give the design. Part 7 gives
the implementation steps, the stages, the tests and the measurements that decide if each stage continues.

Only the literals of 3.2 and the copy of `BacktrackSolver` of step 4 (`TreeSearchSolver`, `LcgSolver`,
`run_search_loop` in `lcg_solver.py`, `lcg_algorithm`) are implemented. The names of the other new arrays and
functions are proposals.

## Contents

- [Why LCG](#why-lcg)
- [Part 1 — The concepts](#part-1--the-concepts)
- [Part 2 — What NuCS has that LCG can use](#part-2--what-nucs-has-that-lcg-can-use)
- [Part 3 — The data structures](#part-3--the-data-structures)
- [Part 4 — The algorithms](#part-4--the-algorithms)
- [Part 5 — Explanations, propagator by propagator](#part-5--explanations-propagator-by-propagator)
- [Part 6 — Search with learning](#part-6--search-with-learning)
- [Part 7 — Stages, tests and measurements](#part-7--stages-tests-and-measurements)
- [Part 8 — Risks and open questions](#part-8--risks-and-open-questions)
- [Reading list](#reading-list)
- [Glossary](#glossary)

## Why LCG

The decision comes from measurements on the 16 medium instances of the MiniZinc Challenge 2026, with a limit of
120 s for each run (October 2026):

| comparison | wins / ties / losses |
|---|---|
| NuCS against Choco (pure CP) | 4 / 10 / 2 |
| NuCS against Gecode | 2 / 12 / 2 |
| NuCS against CP-SAT | 3 / 5 / 8 |
| Chuffed (search annotations, with learning) against NuCS | 8 / 4 / 4 |
| Chuffed `-f` (free search, with learning) against NuCS | 13 / 1 / 2 |
| Chuffed `-f` against CP-SAT | 7 / 5 / 4 |

NuCS is at parity with the pure CP solvers. The gap is to the learning solvers. In 7 problems (mcm, nonogram,
orthorio, rect-euler, saeling, sdn-chain, gcc-benchmark), no pure CP solver finds a solution in 120 s. Chuffed `-f`
finds a solution, or proves that none exists, in all 7. Chuffed is close to pure LCG: it has no LP relaxation and no
large neighbourhood search. Thus the gain comes from learning.

Two more facts from the same run set the direction of this design:

- **Chuffed is not faster per node than NuCS.** On kitchen, Chuffed `-f` explores about 44k nodes/s and NuCS about
  50k nodes/s. The gain comes from a much smaller search tree. Faster nodes in NuCS have never changed a result in
  the benchmarks; a smaller tree does.
- **Learning gives a part of the gain, and the search that uses the learning gives the rest.** Chuffed with the
  model's search annotations wins 8 problems against NuCS. Chuffed with its activity-based search (`-f`) wins 13.

The data is not in the repository. It is in `~/dev/mzn-challenge/2026/` on the author's machine, next to the
models and the instances: the results are `results_medium_*.csv` (the Chuffed arms are `results_medium_chuffed_i.csv`
and `results_medium_chuffed_fi.csv`), the statistics are `stats_medium_*.json`, and the harness is `rerun_arm.py`.
This document gives the other files by name only; they are all in that directory.

## Part 1 — The concepts

This part explains each idea once, first in general and then in NuCS terms. The [reading list](#reading-list) at the
end gives the papers and tutorials, with a suggested reading order. For a first pass, read the CDCL page on
Wikipedia ([cdcl-wiki]), then Stuckey's SAT 2013 slides from slide 46 ([stuckey-sat2013]), then the free CP 2007
version of the main LCG paper ([ohrimenko2007]).

### 1.1 From propagation to learning

A classical CP solver propagates, branches and backtracks. When it finds a failure, it undoes the last decision and
tries the other branch. It does not remember *why* the failure occurred. Thus it can find the same failure again
and again in other parts of the tree. This effect is called **thrashing**.

A SAT solver of the CDCL kind (conflict-driven clause learning) records the cause of each failure as a new
constraint, a **learned clause**. The new clause prevents the same failure in all the rest of the search. CDCL
was introduced by GRASP ([grasp]) and made fast by Chaff ([chaff]) and MiniSat ([minisat]).

**LCG puts CDCL inside a CP solver.** The propagators stay. Each time that a propagator changes a domain, the solver
can tell *which earlier domain changes caused this change*. This record is an **explanation**. With the explanations,
the solver can do conflict analysis as a SAT solver does, and learn clauses over domain changes. The word *lazy*
means that the solver does not translate the constraints into clauses before the search. It makes clauses only when
a propagation needs them ([ohrimenko2009], [feydy2009]).

### 1.2 Literals: facts about domains

A **literal** is a fact that is true or false at a node of the search. In SAT, a literal is a Boolean variable or
its negation. In LCG with integer variables, the literals are facts about the domains:

| literal | meaning | in NuCS terms |
|---|---|---|
| `[x >= v]` | the lower bound of `x` is at least `v` | `domains[x, DOMAIN_MIN] >= v` |
| `[x <= v]` | the upper bound of `x` is at most `v` | `domains[x, DOMAIN_MAX] <= v` |

These are **bound literals**. The negation of `[x >= v]` is `[x <= v - 1]`. A 0/1 variable `b` has the two
literals `[b >= 1]` (true) and `[b <= 0]` (false).

Other LCG solvers also use **equality literals** `[x = v]`, because their domains can have holes. NuCS domains are
intervals (see `ARCHITECTURE.md`, *Interval domains only*). Thus NuCS needs only bound literals. That is a real
simplification, and Part 3 depends on it.

A literal is **true** when the current domains make it true. It is **false** when its negation is true. Otherwise
it is **unassigned**. For example, with `x` in `[2, 7]`: `[x >= 2]` is true, `[x <= 1]` is false, and `[x >= 5]` is
unassigned.

### 1.3 Clauses and nogoods

A **clause** is a disjunction of literals: at least one of them must be true. A **nogood** is a conjunction of
literals that must not all be true. They are the same constraint, written in two ways:

```
nogood:  not ([x >= 2] and [u >= 3])
clause:  [x <= 1] or [u <= 2]
```

This document writes the result of conflict analysis as a nogood, because that is closer to how CP people think
("these facts together cannot hold"). The clause store keeps the clause form, because propagation is easier in that
form.

### 1.4 Decision levels and the implication trail

Each decision opens a new **decision level**. The root is level 0, the first decision is level 1, and so on. In the
LCG mode of NuCS, the decision level is the depth of the choice-point stack (`choice_point_top`), because each level
has one choice point (4.2). This is not true in `BacktrackSolver`: there, an `EQ` decision inside the domain parks
two alternatives and pushes two choice points (`branch` in `choice_points.py`).

The **implication trail** is the list of all the literals that became true, in the order in which they became true.
Each entry records:

- the literal (for example `[x >= 3]`),
- its decision level,
- its **reason**: a decision, a propagator, or a learned clause.

NuCS already has a trail, but it is an *undo* trail. It records the old value of a memory cell, only the first time
that the cell changes at a choice point. LCG needs every change, in order, with its reason. Part 3 adds a second
trail for this.

### 1.5 Explanations

An **explanation** of a literal `l` is a set of literals, all true *before* `l`, which together force `l` through one
constraint. Example: the propagator of `x + y <= 10` sets `[x <= 6]` because `y >= 4`. The explanation is `[y >= 4]`.
It gives the clause `[y <= 3] or [x <= 6]`, which is true in all solutions.

An explanation must be **correct**: the literals of the explanation must force `l` through the constraint alone, in
every state where they hold. An explanation should also be **small**: a short explanation gives short learned
clauses, and short clauses prune more.

There is always a correct explanation, the **naive explanation**: the current bounds of *all* the variables of the
propagator. It is correct for any sound propagator, but it is long. Part 4.3 uses it as the fallback, so that LCG works
with every propagator from the start.

**Lazy explanation** means that the solver does not compute the explanation when the propagator changes the domain.
It only records which propagator made the change. It asks the propagator for the explanation later, during conflict
analysis, and only for the literals that the analysis needs ([feydy2009]). Most changes are never explained. That is
why lazy explanation is cheap.

### 1.6 Conflict analysis and the first UIP

A **conflict** occurs when a propagator finds a failure (a domain becomes empty, or the constraint cannot hold). The
propagator explains the failure with a nogood: a set of true literals that cannot all hold.

Conflict analysis transforms this nogood into a learned clause:

1. Start with the nogood of the failure.
2. While the nogood has more than one literal from the current decision level: take the literal of the current level
   that became true *last*, and replace it with its explanation.
3. Stop when exactly one literal of the current level is left. This literal is the **first unique implication point**
   (first UIP, or 1UIP).

The learned clause is the negation of the final nogood. It has exactly one literal from the current level. Thus,
after the solver goes back to an earlier level, the clause forces the negation of the UIP at once. Such a clause is
an **asserting clause**. The 1UIP rule comes from GRASP and Chaff; the CDCL chapter of the Handbook of
Satisfiability explains it in detail ([cdcl-handbook]).

Replacing a literal with its explanation is **resolution** in SAT terms. In CP terms, it is the question "why was
this fact true?", asked again and again, but only for the current level.

### 1.7 Backjumping

After the analysis, the solver does not undo only the last decision. It goes back to the **assertion level**: the
highest level among the other literals of the learned clause (level 0 if there is none). This is **backjumping**. It
skips the decisions that had no part in the conflict. At the assertion level, the learned clause forces the negation
of the UIP, and propagation continues.

A CDCL solver does not need the "other branch" of a decision. The learned clause *is* the other branch, in a
stronger form. Part 4.5 explains how this changes the choice points of NuCS.

### 1.8 Propagation of learned clauses

A learned clause is a constraint. When all its literals but one are false, the last literal must be true: this is
**unit propagation**. SAT solvers make unit propagation fast with **two watched literals** ([chaff]): each clause
watches two of its literals that are not false. The solver visits a clause only when one of its two watched
literals becomes false. Most clauses are never visited at most nodes.

### 1.9 Activity-based search and restarts

A CDCL solver usually chooses the next decision variable by **activity**. Each time that a variable takes part in a
conflict analysis, its activity increases. All activities slowly decrease (decay), so recent conflicts weigh more.
The solver branches on the most active variable. This is **VSIDS** ([chaff]). Newer variants are LRB ([lrb]) and CHB
([chb]). In CP terms, it is close to dom/wdeg, with two differences: the weight is on variables, not on
constraints, and it comes from the analysis, not only from the propagator that failed.

**Phase saving** chooses the value side: the solver tries the side that the variable had the last time that it was
assigned. **Restarts** go back to the root often, but keep the learned clauses and the activities. With learning,
restarts are cheap, because the learned clauses keep what the search found.

These three parts explain the difference between Chuffed and Chuffed `-f` in the table above.

### 1.10 Clause database reduction

The solver learns one clause at each conflict. After millions of conflicts, the clauses use much memory and slow the
propagation. Thus the solver deletes the clauses that do not seem useful. A good measure is the **LBD** (literal
block distance): the number of distinct decision levels in a clause. Clauses with a low LBD are kept ([glucose]).

### 1.11 A worked example

Variables: `w`, `v` in `{0, 1}`; `x`, `u` in `[0, 5]`; `y` in `[0, 9]`. Constraints:

| name | constraint | propagation used here |
|---|---|---|
| C5 | `w <= v` | `[w >= 1]` forces `[v >= 1]` |
| C6 | `v = 1 -> x >= 2` (half reified) | `[v >= 1]` forces `[x >= 2]` |
| C7 | `v = 1 -> u >= 3` (half reified) | `[v >= 1]` forces `[u >= 3]` |
| C8 | `x + u <= 4` | fails if `[x >= 2]` and `[u >= 3]` |
| C9 | some constraints on `y` | not part of the conflict |

The search makes two decisions:

```
level 1  decision  [y <= 4]          (C9 propagates; nothing below depends on it)
level 2  decision  [w >= 1]
         C5        [v >= 1]          reason: [w >= 1]
         C6        [x >= 2]          reason: [v >= 1]
         C7        [u >= 3]          reason: [v >= 1]
         C8        failure           nogood: [x >= 2] and [u >= 3]
```

Conflict analysis at level 2:

```
nogood                         level-2 literals          step
{[x >= 2], [u >= 3]}           [x >= 2], [u >= 3]        [u >= 3] is the last; replace it with its reason [v >= 1]
{[x >= 2], [v >= 1]}           [x >= 2], [v >= 1]        [x >= 2] is the last; replace it with its reason [v >= 1]
{[v >= 1]}                     [v >= 1]                  one literal of level 2 is left: the first UIP is [v >= 1]
```

The learned clause is `[v <= 0]`. It has no literal from another level, so the assertion level is 0. The solver
**jumps back to the root**, past the decision `[y <= 4]` at level 1, and sets `v = 0` there. For the rest of the
search, `v` is 0, and C5 then sets `w = 0` at the root too. The search never tries `w = 1` again.

Three things in this example are typical:

- **The UIP is not the decision.** The decision was `[w >= 1]`; the analysis found that `[v >= 1]` alone is the
  cause. Any other way to make `v = 1` also fails, and the learned clause covers all of them.
- **The jump skips an unrelated decision.** Classical backtracking would refute `[w >= 1]` at level 2 and continue
  under `[y <= 4]`. After any other choice for `y`, it would try `w = 1` again and find the same failure again.
- **The explanation of C8 is short** because C8 is linear: only the two lower bounds matter. The naive explanation
  of C8 would also contain the upper bounds of `x` and `u` if they had changed. The clause would be longer, and it
  would forbid fewer states.

## Part 2 — What NuCS has that LCG can use

Several decisions that NuCS made for other reasons fit LCG well:

| what NuCS has | why it helps LCG |
|---|---|
| interval domains only | only bound literals are necessary; no equality literals and no lazy creation of Boolean variables |
| `tighten_at` is the only function that writes a domain | one place records each new literal with its reason |
| `bc_algorithm` knows which propagator it runs | the reason of a change is the propagator in the loop at that moment |
| propagators are pure functions on a gathered buffer | a propagator can be called again later on the domains of an earlier node, which is what a lazy explanation needs |
| the failure weights in `weights.py`, with increment growth and rescale | the same mechanism gives the decay of the activities |
| restarts (`restarts.py`) and the restart of the objective bound | CDCL needs restarts; they exist |
| arrays that grow when full (`SOLVER_TRAIL_FULL`, `SOLVER_CHOICE_POINTS_FULL`) | the clause store and the new trail grow in the same way |
| FlatZinc clauses are `SUM_GEQ_C`, `SUM_LEQ_C` or `LINEAR_GEQ_C` over 0/1 variables (`_post_clause`; `SUM_LEQ_C` when all the literals are negative) | the explanation of the linear propagators also explains all the clauses of a model |

Two features need care. The design keeps the first out of the LCG mode; the second fits, for the reasons below:

- **Custom consistency algorithms** (`register_consistency_algorithm`). The LCG mode needs its own propagation loop,
  which records reasons. A custom algorithm cannot record them.
- **Tier B propagator state** (trailed semantic state). These are state cells that a propagator keeps up to date as
  a function of the current domains, and that the engine puts on the trail. See
  [ARCHITECTURE.md](../../ARCHITECTURE.md#propagator-state-a-solver-owned-block-per-propagator). Six propagators use
  it: `lexleq`, `relation`, and `count_eq`, `count_eq_c`, `count_geq_c` and `count_leq_c` at an arity of
  `LIVE_SET_MIN_ARITY` or more. During conflict analysis, the block has the values of the current node, not of the
  node that made the literal. This does not stop the LCG mode from using them (question 7 of 8.2, answered on
  2026-11-02):
  - **The fallback explanation does not call the propagator.** It needs only that the call deduced nothing that the
    box at `REASON_POS` does not imply. The trailed cells hold only facts that the domains of the path made true: a
    variable left the live set, a tuple is dead, a prefix is fixed and equal. The domains only shrink along a path,
    so these facts are still true in the box at the time of the call.
  - **A call with the trailed prefix at zero is correct on any box.** Zero is the root value: the solver allocates
    the block at zero, and `choice_point_init` clears it at each restart. A root call must be correct on any
    initial domains, and the box at `REASON_POS` is only another box. For `count_*` and `relation`, zero selects an
    explicit cold path that builds the live set again from the domains. For `lexleq`, `Q = R = 0` starts the
    automaton at position 0, as the first call does. The debug mode (7.3), the brute-force test (7.2) and an exact
    `explain` function call these propagators on a scratch copy of the block, with the trailed prefix at zero.

## Part 3 — The data structures

### 3.1 A separate solver path

The LCG mode is a separate solver class, `LcgSolver`, with its own compiled step (`run_search_loop`) and its own
propagation loop (`lcg_algorithm`). It shares the propagators, the heuristics, the state, the undo trail, the
choice-point stack and the restarts with `BacktrackSolver`, through their common base class `TreeSearchSolver`.

Reasons:

- **The default path must not get slower.** `SIGN_CONSISTENCY_ALG` and `bc_algorithm` stay as they are. A flag
  tested in `tighten_at` would add arguments and branches to the hottest loop of the solver.
- **No API break.** `BacktrackSolver`, `SIGN_CONSISTENCY_ALG`, `SIGN_COMPUTE_DOMAINS` and the heuristic signatures do
  not change. The new `explain` function of a propagator is an optional argument of `register_propagator`, with the
  fallback as its default. Under the semantic-versioning rule of `bump-version`, the LCG mode is then a minor
  version.
- **The cost is some duplicated code.** The two propagation loops share their inline helpers (`gather`, the event
  scan, `buckets_*`). The duplicated part is the loop itself, about 100 lines.

### 3.2 Literal encoding

A bound literal is a pair `(var_side, value)` of two `int32`:

```
var_side = (variable << 1) | side        side = DOMAIN_MIN (0) for [x >= value], DOMAIN_MAX (1) for [x <= value]
```

This is the same encoding as the flat index of a bound in `state` (`(variable << 1) | bound`). Thus the cell of the
bound that a literal is about is `state[var_side]`, with no translation.

| operation | rule |
|---|---|
| negation of `(x, MIN, v)` | `(x, MAX, v - 1)` |
| negation of `(x, MAX, v)` | `(x, MIN, v + 1)` |
| `(x, MIN, v)` is true | `state[x << 1] >= v` |
| `(x, MAX, v)` is true | `state[(x << 1) \| 1] <= v` |
| `(x, MIN, v)` is false | `state[(x << 1) \| 1] < v` |

NuCS creates no Boolean variable for a literal. Chuffed and CP-SAT create Boolean variables for literals, because
their SAT core works on Boolean variables: Chuffed creates them eagerly for small domains and lazily for large ones
(its `--eager-limit` and `--lazy` options). In NuCS, the clauses work directly on the bounds, so no channelling
between Booleans and bounds is necessary.

### 3.3 The implication trail

The implication trail records each literal that becomes true on the current path:

| array | shape | dtype | holds |
|---|---|---|---|
| `lit_trail` | `(L, LIT_WIDTH)` | int32 | one row for each bound change: `[VAR_SIDE, VALUE, PREV, LEVEL, REASON_KIND, REASON_DATA, REASON_POS]` |
| `lit_top` | `(1,)` | int32 | the number of rows in use |
| `lit_last` | `(2 * domain_nb,)` | int32 | for each `var_side`, the row of its last change, `-1` when the bound has its root value |
| `level_start` | `(H,)` | int32 | for each decision level, the first row of that level |

Column meanings:

- `VAR_SIDE`, `VALUE`: the new bound. The row means "the literal `(VAR_SIDE, VALUE)` became true here". A row also
  makes true every weaker literal on the same bound: after `[x >= 5]`, `[x >= 4]` is true too.
- `PREV`: the row of the previous change of the same `var_side`, or `-1`. The rows of one bound make a chain. The
  value of a bound at an earlier row `t` is the value of the last row of the chain before `t`, or the root value.
- `LEVEL`: the decision level of the row.
- `REASON_KIND`, `REASON_DATA`, `REASON_POS`: see 3.4.

**The new trail is not the undo trail.** The undo trail skips a write to a cell that it already saved at the current
choice point (the write barrier in `design/choice-points.md`). That is correct for an undo log, but conflict analysis needs
every change. The undo trail stays as it is, and restores `state` as before.

**Writes.** Each call of `tighten_at` in the LCG loop adds one row for each bound that it moves: zero, one or two
rows, because `tighten_at` writes `new_min` and `new_max` together. `update_domains` writes the final bounds of one
propagator call, so a call that moves a bound several times internally adds only one row for it.

**Backjumping to level `L`.** The solver pops the rows from `lit_top - 1` down to `level_start[L + 1]`. For each popped
row, it sets `lit_last[VAR_SIDE] = PREV`. The cost is one step for each popped row, the same as for the undo trail.

**Size.** The trail holds only the changes on the current path, so its size is bounded by the number of bound moves
along one branch. It grows by doubling when full, as the undo trail does (a new status `SOLVER_LIT_TRAIL_FULL`).

### 3.4 Reasons

| `REASON_KIND` | `REASON_DATA` | `REASON_POS` | meaning |
|---|---|---|---|
| `REASON_DECISION` | — | — | a decision of the search |
| `REASON_PROPAGATOR` | the propagator | the first row of the propagator call | a propagator made the change |
| `REASON_CLAUSE` | the clause | — | a learned clause made the change |
| `REASON_ROOT` | — | — | a fact that holds for the rest of the search, for example the bound of the best objective (4.7); never explained |

`REASON_POS` is what makes lazy explanations possible. To explain a change, the solver needs the bounds that the
propagator *read* when it made the change. These are the bounds of its variables at row `REASON_POS`. The solver
rebuilds them from the chains (`PREV`). It does not store a copy of the domains for each call.

### 3.5 The learned-clause store

The clauses are in flat arrays, CSR style, as the propagator tables are:

| array | shape | dtype | holds |
|---|---|---|---|
| `clause_lits` | `(C_LITS, 2)` | int32 | the literals of all the clauses, concatenated: `[var_side, value]` |
| `clause_start` | `(C + 1,)` | int32 | for each clause, its first literal; the clause ends where the next one starts |
| `clause_info` | `(C, CLAUSE_WIDTH)` | int32 | `[LBD, ACTIVITY_BUCKET, DELETED]` |
| `watch_head` | `(2 * domain_nb,)` | int32 | for each `var_side`, the first watch of the literals that a change of this bound can make false |
| `watches` | `(W, WATCH_WIDTH)` | int32 | linked watch entries: `[CLAUSE, VALUE, NEXT]` |
| `unit_lits` | `(U, 2)` | int32 | the learned clauses with one literal: `[var_side, value]`; they have no watches (4.7) |

**Which bound wakes which watch.** A literal `[x >= v]` becomes false when the *upper* bound of `x` goes below `v`.
Thus its watch is in the list of `(x, MAX)`. A literal `[x <= v]` becomes false when the lower bound goes above `v`;
its watch is in the list of `(x, MIN)`. The `VALUE` of a watch lets the solver skip a watch without reading the
clause: if the literal is not false yet, the solver goes to the next watch.

**Growth.** Learned clauses are added during the search. The arrays grow by doubling when full, with a new status
`SOLVER_CLAUSES_FULL`, as the other arrays do. Deleted clauses (Part 6.4) are compacted when the solver restarts.

**Why the learned clauses are not propagators.** A clause is a constraint, so it could be a propagator, as the model
clauses of a FlatZinc model are (`_post_clause`). For the learned clauses, the store is better for five reasons:

1. **Wake-up.** The engine wakes a propagator on each `MIN` or `MAX` change of each of its variables. A clause can
   only become unit or fail when one of its literals becomes false: `[x >= 5]` only when `x.max` goes below 5. With
   two watched literals, the solver visits a clause only for its two watches, and the `VALUE` of a watch skips the
   clause with one compare. The events of NuCS have no threshold, so a clause propagator would run on each bound
   change of each of its variables, and most of these calls would do nothing.
2. **The cost of a call.** The engine spends about 110 ns around each propagator call: the queue, the statistics,
   the gather, the dispatch and `update_domains` (`engine-per-call-overhead.md`). A watch visit is a few loads, and a
   CDCL solver makes millions of them.
3. **Static tables.** `Problem` builds `propagator_variables`, `offsets`, `triggers` and the state blocks once. The
   search adds a clause at each conflict, and the reduction (6.4) deletes half of them at a restart. As propagators,
   each addition and each deletion would rebuild these tables and make the trigger lists longer, and the trigger
   lists are on the hottest path of the engine. The store grows and is compacted without a change to these tables.
4. **No work on backtrack.** The two watches of a clause stay correct after a backtrack, so nothing goes on the
   trail. A clause propagator with trailed state would put its state on the trail at each call.
5. **The weights of dom/wdeg.** The weights are kept for each propagator. With learned clauses as propagators, the
   failures of the clauses would go into dom/wdeg, and a variable would get more weight because it is in more
   learned clauses. The activities of 3.6 are kept separately for this reason.

A middle way is one propagator that holds the whole store, as `PropNogoods` does in Choco. In NuCS, that propagator
would need to know which bounds changed, to visit only their watches. NuCS gives no delta to a propagator, and a
delta was measured slower (0.75–0.80× for `count_eq` on magic_sequence, `ARCHITECTURE.md`). The clause queue of
4.1 is a delta for the clauses only, and the propagators do not pay for it.

The model clauses stay propagators: there are few of them, they do not change, and they wake as the other
propagators do. Chuffed and CP-SAT divide the clauses in the same way.

### 3.6 Activities

| array | shape | dtype | holds |
|---|---|---|---|
| `variable_activity` | `(domain_nb + 2,)` | float64 | the activity of each variable, then the increment and its growth |

The layout and the decay are those of `propagator_weights` (`weights.py`): the increment grows by `1 / decay` at each
conflict, and a rescale divides all the cells when the increment passes `1e100`. The activities are not trailed and
not reset at a restart.

## Part 4 — The algorithms

### 4.1 The propagation loop `lcg_algorithm`

`lcg_algorithm` is `bc_algorithm` with three changes:

1. **Before the call of a propagator**, it keeps the current `lit_top` as the `REASON_POS` of the call.
2. **In the write-back**, each moved bound adds a row to `lit_trail` with `REASON_PROPAGATOR`, the propagator and
   the `REASON_POS`.
3. **The clause queue runs first.** Each moved bound whose `watch_head` is not empty goes into a clause queue. Before
   the loop pops the next propagator, it empties the clause queue: it visits the watches of each queued bound and
   propagates the clauses (4.4). Clause propagation is cheap and often fails early, so it has the highest priority,
   as the SAT part has in LCG solvers ([feydy2009]).

When a propagator returns `PROP_INCONSISTENCY`, the loop returns the conflict with the failed propagator and the
current `lit_top`, so that the analysis can ask for the explanation of the failure.

### 4.2 Decisions

In the LCG mode, each decision is **one literal** on its own decision level. The domain heuristics stay as they are.
Their results map to literals:

| heuristic result | decision literal(s) |
|---|---|
| `LE` at `v` | `[x <= v]` |
| `GT` at `v` | `[x >= v + 1]` |
| `EQ` at `v`, `v` = min | `[x <= v]` |
| `EQ` at `v`, `v` = max | `[x >= v]` |
| `EQ` at `v`, other `v` | `[x >= v]` on one level, then `[x <= v]` on the next level |

The choice point keeps its trail mark (`CHOICE_POINT_TRAIL_MARK`). It parks **no alternative**: the learned clause
replaces the refutation (1.7).

**The second literal of an interior `EQ`.** The propagation of `[x >= v]` can change `x` before the solver opens the
next level. Then:

- if `[x <= v]` is already true (`x` is fixed to `v`), the solver opens no second level;
- if `[x <= v]` is false (the propagation raised `x.min` above `v`), the solver opens no second level, and asks the
  heuristics for a new decision.

**The refutation of an interior `EQ`.** A conflict under the second level, with the decision `[x <= v]` as UIP,
gives a clause that asserts `[x >= v + 1]` at the first level. The other side, `[x <= v - 1]`, is not a single bound
literal, so no clause asserts it at once. The search stays complete: a later conflict refutes `[x >= v]` itself.
But the search does not visit the values in the same order as `BacktrackSolver` (question 3 of 8.2).

### 4.3 Explanations

A propagator can give an `explain` function to `register_propagator`:

```python
SIGN_EXPLAIN = int64(
    int32[:, ::1],  # domains: the bounds of the propagator's variables at REASON_POS
    int32[::1],  # parameters
    int64,  # var_idx: the variable of the literal, as an index into the propagator's variables; -1 for a failure
    int64,  # side: DOMAIN_MIN for [x >= value], DOMAIN_MAX for [x <= value]
    int64,  # value
    int32[:, ::1],  # out: the explanation, one row [var_idx, side, value] for each literal
)  # returns the number of rows written in out
```

Rules:

- The explanation must be correct for the **weakest** literal that the analysis asks for. The analysis can ask for
  `[x >= 4]` when the propagator set `[x >= 6]`. A weaker literal often has a shorter explanation.
- The literals of the explanation must be true in `domains`. The engine checks this in the debug mode (7.3).
- With `var_idx == -1`, the function explains a failure: the literals in `out` must not all hold.
- The explanation can use only the bounds at `REASON_POS`, never a bound that the same call set. A propagator that
  loops inside one call must take this into account (5.2).
- The function does not get the hint suffix of the propagator state: the hints do not change the result by
  contract. A function that calls the propagator logic uses a scratch copy of the block, with the trailed prefix
  (Tier B) at zero: Part 2 explains why that call is correct on any box.

**The fallback explanation.** A propagator without `explain` gets `explain_naive`. For each of its variables, it
writes `[x >= min]` if `min` is above the root minimum, and `[x <= max]` if `max` is below the root maximum. This is
correct for every sound propagator. The reason: the propagator removed no solution of its constraint in the box
`domains`. Thus, in every state inside that box, the constraint forces the same literal (or the same failure). The
naive explanation needs no monotonicity and no idempotence. It is only long.

**The cost of a lazy explanation.** To explain a row, the engine rebuilds the bounds of the propagator's variables
at `REASON_POS`, and calls `explain`. This happens only during conflict analysis, and only for the rows that the
analysis visits.

A walk on the `PREV` chains to rebuild these bounds costs the length of each chain, for each variable of the
propagator. For a large linear constraint, or for the objective, whose chain is long, the analysis can then cost
more than the search. The analysis walks `lit_trail` down (4.5), so a cheaper method is possible:

- keep a scratch copy of the bounds, equal to `state` when the analysis starts;
- each time that the walk passes a row, write back into the copy the value that the bound had before the row: the
  `VALUE` of its `PREV` row, or the root value if `PREV` is `-1`;
- to explain a row, also undo in the copy the rows of the same call that are below it (they start at its
  `REASON_POS`), call `explain`, then apply these rows again.

Each row of the trail then costs O(1) for the walk, and each explanation costs the number of rows of its own call.
The first version can use the chains. Change to the scratch copy if the analysis time shows the need.

### 4.4 Clause propagation

For a queued bound `(x, side)`, the solver walks the watch list of `(x, side)`. For each watch:

1. If the watched literal is not false (compare `VALUE` with the bound), go to the next watch.
2. Else, look in the clause for another literal that is not false. If there is one, move the watch to it.
3. Else, if the other watched literal is false, the clause fails: a conflict with `REASON_CLAUSE`.
4. Else, the other watched literal must become true: `tighten_at` sets it, with `REASON_CLAUSE`.

The explanation of a change by a clause is the negation of the other literals of the clause. It needs no `explain`
function.

### 4.5 Conflict analysis

Input: a nogood (from `explain(-1)` of the failed propagator, or from a failed clause). Output: a learned clause and
an assertion level.

```
analyze(nogood):
    level = current decision level
    for each literal in nogood: add(literal)
    walk the rows of lit_trail from lit_top - 1 down:
        if the row makes true a marked literal of the current level:
            if it is the only marked literal of the current level that is left:
                the first UIP is this literal; stop
            unmark it, and add(each literal of its explanation)
    learned clause = negation of the marked literals
    assertion level = highest level among the literals other than the UIP, or 0

add(literal):
    find the row that made the literal true: the FIRST row of the chain of its bound with a value at least as strong
    if the row has level 0, or REASON_ROOT: drop the literal (it is always true)
    merge it with a marked literal of the same bound: keep the stronger one
    mark it
```

Two points are specific to bound literals:

- **The row of a literal is the first row that made it true**, not the last change of the bound. `[x >= 3]` became
  true at the first row of the chain of `(x, MIN)` with a value of at least 3. A later row that set `[x >= 5]` has
  nothing to do with `[x >= 3]`. Using the wrong row gives wrong levels, and wrong assertion levels.
- **Several literals on one bound merge into one.** In a nogood, `[x >= 2]` and `[x >= 3]` together mean `[x >= 3]`.

**A conflict can have no literal of the current level.** Example: the nogood `{[obj >= f]}` of 4.7, when `obj`
became fixed at a lower level. The 1UIP walk above needs at least one literal of the current level. Thus the
solver first backjumps to the highest level among the literals of the nogood, and then does the analysis at that
level. If all the literals of the nogood are at level 0, the problem has no (more) solutions.

The analysis also bumps the activity of every variable that it marks (Part 6).

### 4.6 Backjumping

After the analysis, with assertion level `L`:

1. `trail_undo` to `choice_point_stk[L, CHOICE_POINT_TRAIL_MARK]`: the trail size just before the decision of
   level `L + 1`. This restores `state` to the end of level `L`, as a backtrack does. The mark of the choice point
   that opened level `L` (index `L - 1`) is one level too low: it would also undo the decision of level `L`.
2. Pop the rows of `lit_trail` above `level_start[L + 1]`, and fix `lit_last` (3.3).
3. Set `choice_point_top` to `L`.
4. Add the learned clause to the store, with watches on the asserting literal and on the literal of level `L`. A
   clause with one literal goes into `unit_lits` instead (4.7).
5. Set the asserting literal (the negation of the UIP) with `tighten_at`, with `REASON_CLAUSE`.
6. Propagate.

If the learned clause is empty, or its asserting literal fails at level 0, the problem has no (more) solutions.

### 4.7 Optimization and enumeration

- **Optimization.** When a solution with objective `f` is found (minimization), the bound `[obj <= f - 1]` holds for
  the rest of the search. It cannot be a row of level 0. A write made deep in the tree is on the undo trail, and a
  backtrack removes it. For this reason, `backtrack` today applies the bound again at each choice point that the
  search resumes (`tighten_objective` in `choice_points.py`). The LCG mode resumes no alternative. Thus it applies
  the bound again after each backjump, and after each restart (as `_tighten_objective_at_root` does). Each
  application adds a row with `REASON_ROOT` at the current level. The analysis drops such a row as it drops a
  level-0 fact (4.5), and a backjump below it removes it.

  The current node violates the bound: this is a conflict with the nogood `{[obj >= f]}`. This nogood can have no
  literal of the current level; 4.5 gives the rule for that case. The normal analysis then learns why `obj` was so
  large, and jumps back. This replaces both `OPTIM_RESET` and `OPTIM_PRUNE`: the bound is a fact, and the analysis
  decides how far to go back.
- **Enumeration.** After a solution, the solver adds the nogood of the decision literals of the current path, and
  analyses it as a conflict. Enumeration stays complete, and no solution comes twice. Restarts stop at the first
  solution, as today.
- **Restarts.** A restart is a backjump to level 0. The learned clauses and the activities stay. But the level-0
  facts that the search learned do not stay. `_reset_to_root` (`tree_search_solver.py`) calls `choice_point_init`,
  which writes `problem.initial_domains` back into `state`. The propagation at the root finds again the facts that
  come from the propagators and from the clauses with two or more literals. It does not find the clauses with one
  literal: they have no second watch, and nothing wakes them. Thus the LCG mode keeps them in `unit_lits` (3.5), and
  applies them again after each reset, before the root propagation. Without this, the solver of 1.11 forgets
  `v = 0` at the first restart.

## Part 5 — Explanations, propagator by propagator

The order comes from the calls that NuCS makes on the 7 problems that need learning (`stats_medium_nucs.json`, f3fd7dd;
each problem weighs 1), with one exception: the linear group is at rank 3, above `REGULAR`. One explanation covers
the whole group and every clause of a model (`_post_clause`), and it is the explanation that the literature
documents best:

| rank | propagator | share | used by | explanation |
|---|---|---|---|---|
| 1 | `EQ_C_IMP` (`b -> x = c`) | 1.40 | sdn-chain (97%), mcm (42%) | trivial |
| 2 | `LEQ_C_IMP` (`b -> x <= c`) | 1.19 | rect-euler (81%), saeling (37%) | trivial |
| 3 | `LINEAR_EQ_C`, `LINEAR_GEQ_C`, `SUM_GEQ_C`, `LINEAR_LEQ_C` (and `SUM_LEQ_C`, not in the share) | 0.9 together | saeling, mcm, rect-euler, and every clause of a model | standard linear explanation; `LINEAR_EQ_C` needs both bounds of each term (5.2) |
| 4 | `REGULAR` | 1.00 | nonogram (100%) | hard; fallback first, then an MDD-style explanation ([gange-mdd]) or a decomposition |
| 5 | `LINEAR_NEQ_C` | 0.67 | gcc-benchmark (67%) | simple on bounds |
| 6 | `ELEMENT_L_EQ`, `ELEMENT_EQ` | 0.43 | orthorio (39%) | no dedicated paper; derive it from the decomposition |
| 7 | `MEMBER` | 0.24 | gcc-benchmark | trivial |
| 8 | `EQ_C_REIF`, `LEQ_C_REIF`, `NEQ_C_IMP` | 0.39 together | orthorio, saeling | trivial |
| 9 | `MUL_EQ`, `GCC` | 0.31 together | mcm, gcc-benchmark | fallback first |

### 5.1 Reified and half-reified comparisons with a constant

These are the most called propagators in the 7 problems, and their explanations have one or two literals.

`LEQ_C_IMP`, `b -> x <= c`:

| inference | explanation |
|---|---|
| `[x <= c]` | `[b >= 1]` |
| `[b <= 0]` | `[x >= c + 1]` |
| failure | `[b >= 1]` and `[x >= c + 1]` |

`EQ_C_IMP`, `b -> x = c` (on bounds, `x != c` can be inferred only at a bound):

| inference | explanation |
|---|---|
| `[x >= c]` and `[x <= c]` | `[b >= 1]` |
| `[b <= 0]` | `[x >= c + 1]`, or `[x <= c - 1]`, whichever holds |
| failure | `[b >= 1]` and (`[x >= c + 1]` or `[x <= c - 1]`) |

The full reifications (`_REIF`) add the inferences in the other direction (`[x <= c]` forces `[b >= 1]`, and so on),
each with a one-literal explanation.

### 5.2 Linear inequalities

For `sum(a_i * x_i) <= c`, the propagator sets the upper bound of `x_j` (for `a_j > 0`) from the minimal contribution of
the other terms. The explanation of `[x_j <= v]` is, for each other term `i`:

- `[x_i >= min_i]` if `a_i > 0`,
- `[x_i <= max_i]` if `a_i < 0`,

where `min_i`, `max_i` are the bounds at `REASON_POS`. The terms whose bound is the root bound are dropped (they are
level-0 facts). The failure has the same explanation over all the terms.

The LCG papers explain this case with examples ([ohrimenko2007], examples 9 and 10). Conflict analysis in MIP
solvers uses the same idea ([achterberg]).

**Lifting.** When the slack allows, a weaker bound of some `x_i` still forces the literal. A weaker bound gives a
weaker, more general explanation. That is the *finesse* option of Chuffed. The first version does not lift; a later
stage can measure it.

`SUM_GEQ_C`, `SUM_LEQ_C` and the clauses of FlatZinc (`_post_clause`) are the case with all `a_i = 1` over 0/1
variables. For a clause, the explanation is exactly the other literals of the clause.

**`LINEAR_EQ_C` and the propagators that loop inside one call.** `LINEAR_EQ_C` is two inequalities, but its
propagator does not apply them one after the other. It loops inside one call (`linear_eq_c_propagator.py`) until
neither side changes a bound. Thus it can raise `min_k` with the `>=` side, and then use the new `min_k` in the
`<=` side to lower `max_j`. The trail gets one row for each bound at the end of the call, and an explanation can
use only the bounds at `REASON_POS` (4.3). The new `min_k` is not true at `REASON_POS`. Thus the rule above, with
one bound for each other term, is **not correct** for `LINEAR_EQ_C`: the explanation of `[x_j <= v]` must also
contain the literals that raised `min_k`, which are upper bounds of the other terms. There are two options:

- explain from the box at `REASON_POS`, with both bounds of each term that the loop used: correct, but longer;
- record a row for each step of the loop, each with its own `REASON_POS` inside the call: shorter explanations,
  but more rows and a change to `update_domains`.

Stage 2 starts with the first option, and measures the mean length of the clauses before it tries the second. The
same problem applies to every propagator that loops inside one call. The brute-force test of 7.2 finds this error.

**A third option: remove the loop.** Six propagators are idempotent because they loop inside one call:
`linear_eq_c`, `mul_eq`, `element_l_eq_alldifferent`, `gcc`, `circuit_chains` and `subcircuit`. Without the loop, a
propagator does one pass and is registered with `idempotent=False`, so the engine calls it again after each call
that changed a bound. Each call then reads only the bounds at its `REASON_POS`, and a one-step explanation is
correct. The cost of this change was measured on 2026-11-02 (base `7bf10c7`), with the protocol of
`design/benchmarking.md`: each variant against the base, 5 to 10 repetitions in random order, a random-length
environment variable for each run, and `queens` as the control. Each FlatZinc model ran for a fixed budget of
failures, so both builds did the same work. All the trees were the same in both builds.

| propagator | subject | calls of the propagator | time, without / with the loop |
|---|---|---|---|
| `linear_eq_c` | kitchen | +78% | **1.30** |
| `linear_eq_c` | atp-stage2 | +66% | **1.13** |
| `linear_eq_c` | gbac | +8% | **1.08** |
| `linear_eq_c` | tdtsp | +21% | 1.00 |
| `gcc` | sports `-n 10`, `SMALLEST_DOMAIN` | +16% | **1.08** |
| `gcc` | gcc-benchmark | +0.3% | 0.98 |
| `mul_eq` | nside | +0.1% | 0.98 |
| `mul_eq` | gbac | +1% | 0.97 |
| `mul_eq` | multiple-constant-multiplication | +77% | 1.01 |
| `element_l_eq_alldifferent` | quasigroup `-n 12`, `SMALLEST_DOMAIN` | +2% | 0.99 |
| `circuit_chains` | tsp gr24 | +15% | 0.98 |
| `circuit_chains` | atp-stage2 | +0.3% | 1.00 |
| `subcircuit` | orthorio | +4% | 1.00 |

The loop pays when the propagator often changes a bound, because the engine round trip of the extra call costs
more than one more pass inside the call. Thus:

- `linear_eq_c` and `gcc` keep their loop, and use the first option above.
- `mul_eq`, `element_l_eq_alldifferent`, `circuit_chains` and `subcircuit` can do one pass in the LCG mode: no
  difference is above the 5% floor of `design/benchmarking.md`. Their explanations then explain one pass only.
  For `circuit_chains`, this keeps the one-rule explanations of [francis-circuit].

With dom/wdeg, the trees were not the same: the order of the propagation decides which propagator gets the failure
weight. On quasigroup and sports with their default dom/wdeg search, the trees without the loop were 2 and 3 times
smaller. This is an effect of the heuristic, not of the cost of the loop, and the table does not use these runs.

### 5.3 The other propagators of the list

- **`LINEAR_NEQ_C`**, `sum(a_i * x_i) != c`: on bounds, it infers only when all the variables but one are fixed and
  the forbidden value is at a bound of the last one. The explanation: the fixed values of the others (two literals
  each), and the bound of the last variable that touches the forbidden value.
- **`MEMBER`**, `x in S`: it moves a bound of `x` past the values that are not in `S`. The explanation of
  `[x >= w]` is `[x >= s + 1]`, where `s` is the largest value of `S` below `w`. If no value of `S` is below `w`,
  the constraint alone forces the literal, and the explanation is empty. The bound before the move would also be
  correct, but it is stronger, so the learned clause would be less general. The same rule explains the upper bound.
- **`ELEMENT_EQ` / `ELEMENT_L_EQ`**, `z = a[i]`: the bounds of `z` come from the bounds of the elements in the range
  of `i`, and the bounds of `i` come from the elements that cannot equal `z`. The explanation: the bounds of `i`, and
  the bounds of the elements that the inference used. There is no dedicated paper; write the explanation from the
  decomposition of `element` into implications (`i = k -> z = a[k]`), and check it with the brute-force test.
- **`REGULAR`**: the propagator works on a layered graph of the automaton, which is an MDD. The explanation of MDD
  propagators is in [gange-mdd]. The exact explanation follows the paths that the bounds cut. Start with the fallback, and measure nonogram. If the fallback is too
  weak, compare an exact explanation with a decomposition of `regular` into clauses at model build.
- **`MUL_EQ`, `GCC`, `ALLDIFFERENT`**: fallback first. The bounds-consistent `alldifferent` has a known explanation
  based on Hall intervals ([downing-alldiff]).
- **The globals where NuCS is strong**: `circuit_chains`, `disjunctive`, `cumulative`. NuCS wins atp-stage2 and
  tdtsp with them. In the LCG mode, they keep their propagation, with the fallback explanation at first. Later
  stages can add the explanations of the literature: circuit ([francis-circuit]), the unary resource, which is
  `disjunctive` ([vilim-unary]), and cumulative ([schutt-cumulative], [schutt-ttef]).

## Part 6 — Search with learning

### 6.1 Activity

A new variable heuristic `VAR_HEURISTIC_ACTIVITY` chooses the unbound decision variable with the highest activity.
Ties go to the smallest domain, then to the first variable. The conflict analysis bumps the activity of each
variable that it marks (4.5).

`SIGN_VAR_HEURISTIC` gives the heuristics `propagator_weights` and no variable activities. Two options, to decide
in stage 3 (Part 8, question 2):

- add the activities after the weights in the same `float64` array, which keeps the signature;
- add a new argument, which changes the signature (a major version).

### 6.2 Value choice

Phase saving, adapted to bounds: for each variable, the solver keeps the value that it had in the last solution, or
when it was last bound. The new domain heuristic `DOM_HEURISTIC_SAVED_VALUE` splits toward that value: `EQ` at the
saved value, or the nearest bound. Without a saved value, it splits at the middle. This is close to
solution-based phase saving, which Chuffed offers as `--sbps`.

### 6.3 Restarts and the free search of `fzn-nucs`

The restarts exist (`restarts.py`). With learning, they help more than without it: each descent starts with all
that the earlier descents learned. `fzn-nucs -f` uses dom/wdeg, last-conflict and Luby 500 today. In the LCG mode,
`-f` becomes activity, saved values and Luby restarts. The scale of the restarts is a parameter to measure.

### 6.4 Clause database reduction

At each restart, if the store holds more than a limit of clauses, the solver deletes half of the clauses with the
highest LBD. The limit grows slowly with the number of conflicts ([glucose]). SAT solvers keep the clauses that are
the reason of a row of the trail. At a restart, only the rows of level 0 are left, and the analysis never explains
them (4.5). Thus no clause is a reason at a restart, and the reduction needs no such test. A reduction at another
time would need it. The first version can keep all the clauses, and add the reduction when memory or propagation
time shows the need.

## Part 7 — Stages, tests and measurements

### 7.0 Implementation steps

The steps below divide the stages of 7.1 into commits or PRs. Each step has a check that must pass before the next
step starts. `BacktrackSolver` does not change at any step.

Steps 4 to 9 do not change the search. While the tree stays the same, the best test of a new mode is that it gives
*the same* counters as the old mode. Step 10 is the first step that changes the tree. At that step, the trail, the
explanations, the analysis and the store already have their tests.

#### Phase 0 — Preparation

1. **Answer the open questions that block the work.** All are answered except question 2 of 8.2 (the activities),
   which waits until Phase 4. The benchmark data is in `~/dev/mzn-challenge/2026/` ([Why LCG](#why-lcg)). Before
   a run of `rerun_arm.py`, check that `minizinc --solvers` lists the current NuCS version: the harness calls
   `minizinc --solver nucs`, so a registration that points to an old install measures old code.
2. **Record a baseline.** Run the current `BacktrackSolver` on the 16 medium instances and on the examples. Record
   the nodes, the time and the results. Choose a control model for the A/B runs (7.4).
   *Done on 2026-11-03, at commit 613c74d:* the files and the control are in 7.4.

#### Phase 1 — Record the reasons, with the search unchanged

3. **Add the literals module.** It holds the encoding `(var_side, value)`, the negation, and the tests "true" and
   "false" on `state` (3.2).
   *Check:* unit tests.
   *Done on 2026-11-03:* `nucs/solvers/lcg/literals.py` (`lit_negate`, `lit_is_true`, `lit_is_false`, all
   `inline="always"`, because `lcg_algorithm` will be compiled with `_nrt=False` as `bc_algorithm` is) and
   `tests/solvers/lcg/test_literals.py` (curated cases and a brute-force test on all the boxes in `[-3, 3]`).
4. **Add an `LcgSolver` that is a copy of `BacktrackSolver`.** It gets its own `run_search_loop` and its own
   `lcg_algorithm`, which are exact copies of the current code at this step.
   *Check:* on all the examples, every counter is the same as with `BacktrackSolver`. Each later step then shows as
   a small diff against this copy.
   *Done on 2026-11-04:* `TreeSearchSolver` (`nucs/solvers/tree_search_solver.py`) is the new base class of
   `BacktrackSolver` and `LcgSolver` (`nucs/solvers/lcg/lcg_solver.py`). It holds the Python part of the search: the
   state, the restarts, the optimization modes, the growth of the arrays and the time accounting. Its only abstract
   method is `_run_search_loop`, and each subclass gives its compiled loop there. `BacktrackSolver` adds only its
   consistency algorithm. `LcgSolver` adds nothing else: it has no `consistency_algorithm` parameter, because
   `run_search_loop` calls `lcg_algorithm` directly, not through the table of consistency algorithms. Thus the rule
   of step 7 that `LcgSolver` takes no custom consistency algorithm is already true. `lcg_algorithm` is not registered
   as a consistency algorithm: from step 5, its signature is not `SIGN_CONSISTENCY_ALG`. A later step that must change
   the shared Python part adds a hook to `TreeSearchSolver` (for example, the growth of a new array at step 5), and a
   subclass never overrides a concrete method. The compiled code of `BacktrackSolver` does not change. All the LCG code
   is in the subpackage `nucs/solvers/lcg/`, and its tests are in `tests/solvers/lcg/`: no other module of `nucs`
   imports it, so `BacktrackSolver` cannot depend on it. The modules of `BacktrackSolver` are in the subpackage
   `nucs/solvers/backtrack/`, and what the two solvers share is at the top of `nucs/solvers/`: `tree_search_solver.py`,
   `search_loops.py` (the statuses and the search control) and `propagation_steps.py` (the steps of one propagator call,
   which `lcg_algorithm` took from `bc_algorithm.py`). The move of the modules of `BacktrackSolver` changes their import
   paths: it is a major version. On the 22 examples at their default arguments (golomb with
   `--consistency-algorithm BC`), every counter is the same as with `BacktrackSolver`, which itself gives the counters
   of the baseline (7.4). magic_square is the exception that 7.4 expects: its unseeded `RANDOM_VALUE` changes the tree,
   and only its solution count is the same.
5. **Add the implication trail** (3.3, 3.4). The arrays `lit_trail`, `lit_top`, `lit_last` and `level_start`. The
   status `SOLVER_LIT_TRAIL_FULL`, with the array doubled when full. The write-back of `lcg_algorithm`, the
   decisions and the objective bound write the rows, and a backtrack pops them. The search still uses the parked
   alternatives.
   *Check:* a debug function replays `lit_trail` from the root and compares the result with `state` at each node.
   The counters stay the same as in step 4. Measure the drop in nodes per second.
6. **Add the decisions as literals** (4.2). Each level has one literal and one choice point. An interior `EQ` uses
   two levels, with the rules of 4.2. The alternatives stay parked.
   *Check:* the solution counts and the optima are the same as with `BacktrackSolver`.
7. **Add the explanations** (4.3). The signature `SIGN_EXPLAIN`, and the optional `explain` argument of
   `register_propagator`. `explain_naive`, and the rebuild of the bounds at `REASON_POS` from the chains.
   `LcgSolver` takes no custom consistency algorithm (true since step 4). It accepts the Tier B propagators (Part 2): the debug mode
   and the tests call them on a scratch copy of the block, with the trailed prefix at zero.
   *Check:* the brute-force explanation test of 7.2, run on `explain_naive` for every propagator. The debug mode of
   7.3.

#### Phase 2 — Learn the clauses (stage 1)

8. **Run the 1UIP analysis in shadow mode** (4.5): the first-row lookup, the merge of literals on the same bound,
   the case with no literal of the current level, and an explicit stack instead of recursion. The classical search
   calls the analysis at each failure, but does not use its result.
   *Check:* the hand-built trail of 1.11 gives `[v <= 0]` at level 0. On small models, each clause that the shadow
   analysis learns is valid against the solutions that brute force finds. In shadow mode, a clause can be checked
   while it has no effect on the search, so an error in the analysis shows before it can remove a solution.
9. **Add the clause store** (3.5, 4.4). The arrays `clause_lits`, `clause_start`, `clause_info`, `watch_head`,
   `watches` and `unit_lits`. The status `SOLVER_CLAUSES_FULL`. The clause queue, which `lcg_algorithm` empties
   before it pops the next propagator. The `REASON_CLAUSE` rows, and the explanation of a row that a clause set.
   *Check:* unit tests on clauses that the test adds by hand. In shadow mode, the learned clauses are stored and
   propagated, but the search still backtracks in the classical way.
10. **Add the backjumps** (4.6). This step changes the search. The learned clause replaces the parked alternatives,
    with the mark at index `L`. An empty clause means that there is no (more) solution.
    *Check:* the solution counts and the optima are the same as with `BacktrackSolver`, with the JIT on and with the
    JIT off. The learned clauses are valid.
11. **Add enumeration, optimization and restarts** (4.7). Enumeration: the blocking nogoods. Optimization: the
    objective bound is applied again after each backjump. Restarts: `lit_trail` is reset, and `unit_lits` is applied
    again after each `_reset_to_root`.
    *Check:* the clause test of 7.2, with its three cases. The examples with restarts.
12. **Add the statistics.** The conflicts, the mean length of the learned clauses, the mean LBD, and, for each
    propagator, the number of rows that the analysis explains.
13. **Make the mode available in FlatZinc.** An option of `fzn-nucs` selects `LcgSolver`. Run the differential tests
    against Gecode and Chuffed on small random instances (7.2).
14. **Gate 1** (7.1).

#### Phase 3 — Exact explanations (stage 2)

15. **Write one `explain` function for each line of the Part 5 table, in its order.** First `EQ_C_IMP` and
    `LEQ_C_IMP`. Then the linear group, with both bounds of each term for `LINEAR_EQ_C` (5.2). Then `REGULAR`
    (measure the fallback on nonogram first), `LINEAR_NEQ_C` and the rest of the table. Each function has its own
    brute-force test and its own measurement of the clause length. The count of explained rows from step 12 can
    change the order: a propagator that runs often but seldom sets a literal that the analysis visits gains little
    from an exact explanation.
16. **Gate 2** (7.1).

#### Phase 4 — Search (stage 3)

17. **Add the activities** (3.6, 6.1). The array `variable_activity` (answer question 2 of 8.2 first), the bump in
    the analysis, and the decay. Then `VAR_HEURISTIC_ACTIVITY`.
18. **Add the saved values** (6.2): `DOM_HEURISTIC_SAVED_VALUE`.
19. **Change the free search** (6.3). `fzn-nucs -f` in the LCG mode uses activity, saved values and Luby restarts.
    Measure the scale of the restarts.
20. **Gate 3** (7.1).

#### Phase 5 — Scale (stage 4)

21. **Reduce the clause store** (6.4): delete the clauses by LBD at a restart, then compact the store.
22. **Explain the strong globals** (5.3): `circuit_chains`, `disjunctive` and `cumulative`.
23. **Do the optional work only if the measurements show the need:** clause minimization (question 6 of 8.2), the
    scratch copy of the bounds for the explanations (4.3), and the sorted watches for large domains (8.1).
24. **Gate 4** (7.1). Also measure the memory on long runs.

#### Phase 6 — Release (stage 5)

25. **Write the documentation.** `ARCHITECTURE.md`, an `explain` step in the `add-propagator` skill, and the
    changelog.
26. **Release.** Run `bump-version` (a minor version if no signature changed). If gate 3 passed, make LCG the
    default of `fzn-nucs -f` (question 5 of 8.2); the annotated search keeps `BacktrackSolver`.

### 7.1 Stages

Each stage ends with a measurement. The next stage starts only if the gate passes. The gates use the 16 medium
instances (120 s), with the same harness as the benchmark above (`rerun_arm.py`), and the examples of `nucs/examples`.

| stage | content | gate |
|---|---|---|
| 1. Infrastructure | `LcgSolver`, `lit_trail` and reasons, decisions as literals, fallback explanations only, 1UIP analysis, backjumping, clause store with two watched literals. The search heuristics stay as they are. | all the tests of 7.2 pass; on most of the instances that both runs solve, the search tree is smaller than without LCG; on the other instances, the best objective or the time to the first solution is better; the number of nodes per second stays above half of the current rate |
| 2. Exact explanations | the propagators of Part 5, in the order of the table | the learned clauses are shorter (mean length); the results move toward Chuffed with annotations (8 / 4 / 4 against NuCS) |
| 3. Search | activity, saved values, restarts; `fzn-nucs -f` in the LCG mode | the results move toward Chuffed `-f` (13 / 1 / 2 against NuCS) |
| 4. Scale | clause database reduction; explanations for `circuit_chains`, `disjunctive`, `cumulative` | no loss on atp-stage2 and tdtsp; the memory stays bounded on long runs |
| 5. Release | documentation (`ARCHITECTURE.md`, the `add-propagator` skill: the `explain` step), changelog | — |

Stage 1 is the largest. Its result is also the most informative: if a smaller tree does not appear with naive
explanations on problems like sdn-chain (where 97% of the calls go to a propagator whose exact explanation has one
literal), the cause must be found before stage 2.

### 7.2 Tests

The tests follow the `write-tests` skill: oracles where bugs hide in corners, and solver-level counts.

- **Explanations, by brute force** (one test for each `explain` function). For random small boxes: run the
  propagator; for each literal that it sets, get the explanation; build the box of the root domains cut by the
  explanation; enumerate all the tuples of that box that satisfy the constraint; check that each one satisfies the
  literal. For a failure, check that no tuple satisfies the constraint. This extends
  `PropagatorTest.assert_sound_against_brute_force`.
- **Learned clauses are valid.** On small models whose solutions can be enumerated without LCG, run the LCG search
  and check each learned clause against the solutions that it must keep:
  - in a search for the first solution, every solution;
  - during enumeration, only the solutions not found yet when the clause was learned, because the clauses that come
    from the blocking nogoods (4.7) remove the solutions found before;
  - during optimization, only the solutions better than the best one when the clause was learned, because the
    clauses that come from the objective bound remove the others.
- **Same answers.** The number of solutions with and without LCG is the same on the examples (queens, magic_square,
  langford, and so on). The optimum with and without LCG is the same (golomb, knapsack, tsp gr17, jobshop mt06).
- **Without the JIT.** The suite runs with `NUMBA_DISABLE_JIT=1` (the CI does it on each push).
- **Differential tests against Gecode and Chuffed** on random small FlatZinc instances, as for the index-set bugs.

### 7.3 A debug mode for explanations

A wrong explanation gives a wrong learned clause, and a wrong learned clause removes solutions. The answer is then
wrong, and nothing shows it. This is the largest risk of LCG.

The debug mode checks each explanation when the analysis asks for it. It rebuilds the box of the explanation, and
calls the propagator on it until nothing changes. One call is not sufficient, for two reasons: some propagators are
not idempotent (`idempotent=False` in `register_propagator`), and the box of an explanation is larger than the box at
`REASON_POS`, so one call can set a weaker literal.

- If the propagator sets the literal (or fails), the explanation is correct.
- If it does not, the explanation can still be correct: an exact explanation can use reasoning that the propagator
  does not do. On a small box, the debug mode then enumerates the tuples, as the brute-force test of 7.2 does, and
  decides. On a large box, it reports the explanation as suspicious, not as wrong.

The tests run with this mode on. The production runs run with it off.

### 7.4 Measurements

Follow the method of the earlier performance work:

- Measure the solve time and the wall time separately (the Numba compile cost is per process).
- Use `BacktrackSolver` on the same instance as the control (see below). The order of runs moves the results by
  about 5%.
- Count the nodes, the conflicts, the mean length of the learned clauses and the mean LBD. The time alone does not
  show why a stage helps or not.
- Compare with Chuffed with the same flags (`MZN_SOLVER_PATH` must point to the solvers of the MiniZinc IDE bundle).

**The control.** Rule 5 of `design/benchmarking.md` asks for a control model that the change cannot affect, and says
that no model is safe for a change to the engine. LCG is a change to the engine, but `BacktrackSolver` does not
change at any step of 7.0. Thus the control is the same instance, solved by `BacktrackSolver`, in the same session,
with the two solvers in alternate order:

```
BacktrackSolver  instance    control
LcgSolver        instance    subject
LcgSolver        instance    subject
BacktrackSolver  instance    control
```

- **The same shape.** The control has the same propagators, domains and memory use as the subject. A different
  model can drift in a different way, for example under cache pressure. The same instance cannot.
- **The ratio is the result.** The gates of 7.1 compare LCG with `BacktrackSolver` (for example, the nodes per
  second). When the two run in the same session, the drift of the machine goes out of the ratio. If the control moves
  as much as the subject between two builds, discard the comparison (rule 6 of `design/benchmarking.md`).

Some steps change code that the two solvers share: for example, step 7 adds `SIGN_EXPLAIN` and the `explain` argument
of `register_propagator`. At these steps, the control is also a regression test:

- Its counters must be *the same* as in the baseline of step 2. The counters do not change from run to run: the
  baseline gave the same choice counts as the run of 2026-09-19 on the two medium instances whose search ends
  (atp-stage2: 8,775,827; tdtsp: 2,109,982).
- Its time must stay inside the drift of the order of runs. If it moves more, the shared change has a cost for
  `BacktrackSolver`, which stays the default solver.
- Clear the Numba cache before such a run. The cache keys on the date of the file of the function only (rule 4 of
  `design/benchmarking.md`), so a stale cache can show an unchanged control.

**The baseline of step 2** (commit 613c74d, 2026-11-03). The files are in `~/dev/mzn-challenge/2026/`:

| file | content |
|---|---|
| `results_medium_nucs_613c74d.csv`, `stats_medium_nucs_613c74d.json` | the 16 medium instances, 120 s, annotated search: the reference of gates 1 and 2 |
| `results_medium_nucs_613c74d_f.csv`, `stats_medium_nucs_613c74d_f.json` | the same with `-f`: the reference of gate 3 |
| `baseline_examples_613c74d.jsonl` | all the counters of the 22 examples at their default arguments |
| `baseline_arm.py`, `baseline_examples.py` | the scripts that wrote them |

- `baseline_arm.py` calls `rerun_arm.py` and adds a tag to the names of the files. Without flags, `rerun_arm.py`
  writes over `results_medium_nucs.csv`, the run of 2026-09-19.
- bibd has no default arguments: the baseline uses the instance of the README (`-v 8 -b 14 -r 7 -k 4 -l 3
  --find-all`, 92 solutions, 712 choices).
- magic_square uses `RANDOM_VALUE` with no seed: its counters change from run to run. Compare only its solution
  count.
- Most of the examples end in less than 1 ms. They are good for the checks of equal counters (steps 4 to 6), but not
  to measure the nodes per second (step 5): for that, use tsp (45,208 choices), langford (18,632), golomb (11,241)
  and the medium instances.
- In the annotated arm, 11 of the 16 medium instances stop at the time limit. For them, gate 1 compares the best
  objective, not the size of the tree. Only atp-stage2, tdtsp and filters end their search.

## Part 8 — Risks and open questions

### 8.1 Risks

| risk | effect | answer |
|---|---|---|
| a wrong `explain` function | wrong answers that look correct | brute-force tests for each `explain`; the debug mode; differential tests |
| naive explanations are too long | the learned clauses prune little; memory grows | exact explanations in the order of Part 5; clause reduction |
| the cost per node in the LCG mode | fewer nodes per second | the gate of stage 1; Chuffed shows that a node rate close to the current one is enough |
| watch lists on large domains | a bound of a large-domain variable can have many watches with different values | sort the watches of a bound by value, or keep a blocking value; measure first |
| Numba limits | no dynamic lists in nopython code; no recursion for clause minimization | flat arrays with growth statuses, as today; an explicit stack instead of recursion |
| scope | LCG touches the loop, the search, the optimization and every propagator | the stages; the fallback explanation keeps every propagator working at every stage, also the six with Tier B state (Part 2) |

### 8.2 Questions and decisions

The decisions below were taken on 2026-11-02. Only question 2 is still open.

1. **A separate solver class, or a mode of `BacktrackSolver`?** This design proposes a separate class (3.1), for a
   default path with no cost and no API break.
   **Decision:** a separate class, `LcgSolver`.
2. **Activities in the weights array, or a new argument?** (6.1). The first keeps `SIGN_VAR_HEURISTIC`; the second is
   clearer but is a major version.
   **Open:** decide in stage 3, when step 17 of 7.0 writes the activity heuristic.
3. **`EQ` decisions as two levels** (4.2). Chuffed uses equality literals for this. Two bound literals on two levels
   is correct, but it doubles the levels for value-by-value searches. Also, a learned clause can refute only one
   side of `x = v` at once (4.2), so the search visits the values in a different order.
   **Decision:** stage 1 uses two levels. Gate 1 measures sdn-chain and mcm, the value-by-value searches, and
   compares the levels and the conflicts with Chuffed. The design changes only if they show a problem.
4. **`regular`: explanation or decomposition?** (5.3).
   **Decision:** the fallback explanation first. Stage 2 measures nonogram. If its clauses stay long, compare the
   MDD-style explanation ([gange-mdd]) with a decomposition into clauses at model build.
5. **Should the LCG mode become the default of `fzn-nucs`** when it wins on the 16 medium instances? And of `-f`
   only, or also of the annotated search?
   **Decision:** the default of `-f` only (6.3). The annotated search keeps `BacktrackSolver`, so it still follows
   the search of the model exactly. A flag of `fzn-nucs` selects LCG for the annotated search.
6. **Clause minimization** (removing the literals that the other literals of the clause already imply) is standard in
   SAT solvers ([minisat]). It is not in the stages above.
   **Decision:** stage 4, and only if the mean length of the clauses stays high after the exact explanations of
   stage 2.
7. **The six propagators with Tier B state** (Part 2): `lexleq`, `relation` and four `count_*`. Is a call with the
   block reset to zero a correct explanation for each, or does each need an `explain` function from stage 1?
   **Answer:** a call with the trailed prefix at zero is correct for all six, and the fallback explanation does not
   need that call at all (Part 2 gives the reasons). The LCG mode accepts them from stage 1, and none of them needs
   an `explain` function to be correct.

## Reading list

All the links were checked on 2026-10-02. "Free" means a free copy; the other links go to the publisher by DOI.

### Start here (in this order)

1. [Wikipedia, *Conflict-driven clause learning*][cdcl-wiki]: a short worked example of an implication graph, a
   learned clause and a backjump. Read it first.
2. [Stuckey, *There are no CNF problems*, SAT 2013 invited talk, slides][stuckey-sat2013] (free): from slide 46, LCG
   as "propagation with learning", for people who know propagation.
3. [Ohrimenko, Stuckey, Codish, *Propagation = Lazy Clause Generation*, CP 2007][ohrimenko2007] (free): the first LCG
   paper. Examples 9 and 10 explain a linear constraint.
4. [Stuckey, *Search is Dead, Long Live Proof*, PPDP 2013, slides][stuckey-ppdp2013] (free): 39 of its 68 slides are
   about explanations and nogoods. The CP-SAT documentation calls it a complete presentation of its technology.
5. [Marques-Silva, Lynce, Malik, *Conflict-Driven Clause Learning SAT Solvers*, Handbook of Satisfiability,
   chapter 4, 2009][cdcl-handbook] (free, first edition): the best single survey of conflict analysis, watched
   literals, restarts, heuristics and clause deletion. Use it as the reference for Parts 4 and 6.

A shorter student lecture that goes from CP to SAT to LCG: [Lal, *From CP to SAT*, UPenn CIS 1921, 2024][lal-2024]
(free).

### CDCL foundations

- [Marques-Silva, Sakallah, *GRASP*, IEEE Transactions on Computers 48(5), 1999][grasp]: the origin of conflict
  analysis, UIPs and backjumping. Free conference version: [ICCAD 1996][grasp-free].
- [Moskewicz, Madigan, Zhao, Zhang, Malik, *Chaff*, DAC 2001][chaff]: two watched literals and VSIDS.
  Free: [PDF][chaff-free].
- [Eén, Sörensson, *An Extensible SAT-solver* (MiniSat), SAT 2003][minisat]: the clearest blueprint of a CDCL loop,
  and the base of the first LCG solvers. Free: [extended version][minisat-free].

### LCG

- [Ohrimenko, Stuckey, Codish, *Propagation via lazy clause generation*, Constraints 14(3), 2009][ohrimenko2009]: the
  journal version of the main paper.
- [Feydy, Stuckey, *Lazy Clause Generation Reengineered*, CP 2009][feydy2009]: the architecture of a SAT engine inside
  a CP solver, with lazy explanations. It is the closest to this design.
- [Stuckey, *Lazy Clause Generation: Combining the Power of SAT and CP (and MIP?) Solving*, CPAIOR 2010][stuckey-cpaior2010]:
  a 5-page overview.
- [Chu, *Improving Combinatorial Optimization*, PhD thesis, University of Melbourne, 2011][chu-thesis] (free): by the
  author of Chuffed.
- [Lecoutre, Saïs, Tabary, Vidal, *Recording and Minimizing Nogoods from Restarts*, JSAT 1, 2007][lecoutre-jsat]: the
  CP way to nogoods without explanations (the "stage 1" that the gate data made less useful). Free: the shorter
  [IJCAI 2007 version][lecoutre-ijcai].

### Explanations of constraints

- Linear: [ohrimenko2007] (examples 9 and 10), and [Achterberg, *Conflict analysis in mixed integer programming*,
  Discrete Optimization 4(1), 2007][achterberg].
- Alldifferent: [Downing, Feydy, Stuckey, *Explaining alldifferent*, ACSC 2012][downing-alldiff], and
  [*Explaining Flow-Based Propagation*, CPAIOR 2012][downing-flow].
- Regular and MDDs: [Gange, Stuckey, Szymanek, *MDD propagators with explanation*, Constraints 16(4), 2011][gange-mdd].
- Unary resource (`disjunctive`): [Vilím, *Computing Explanations for the Unary Resource Constraint*, CPAIOR
  2005][vilim-unary] (free).
- Cumulative: [Schutt, Feydy, Stuckey, Wallace, *Explaining the cumulative propagator*, Constraints 16(3),
  2011][schutt-cumulative], and [Schutt, Feydy, Stuckey, *Explaining Time-Table-Edge-Finding Propagation*, CPAIOR
  2013][schutt-ttef].
- Circuit: [Francis, Stuckey, *Explaining circuit propagation*, Constraints 19(1), 2014][francis-circuit] (free). It
  is directly about the rules that `circuit_chains` uses.
- Element: no dedicated paper was found.

### Search and the clause database

- [Liang, Ganesh, Poupart, Czarnecki, *Learning Rate Based Branching Heuristic for SAT Solvers* (LRB), SAT
  2016][lrb] (free): compares LRB, VSIDS and CHB.
- [Liang et al., *Exponential Recency Weighted Average Branching Heuristic for SAT Solvers* (CHB), AAAI 2016][chb]
  (free).
- [Audemard, Simon, *Predicting Learnt Clauses Quality in Modern SAT Solvers*, IJCAI 2009][glucose] (free): the LBD
  and the clause reduction of Glucose.

### Solver code

- [Chuffed][chuffed]: the reference LCG solver, in C++, MIT license. Read it for the details that the papers leave out.
- [OR-Tools CP-SAT, README][cpsat-readme]: CP-SAT describes itself as a CP solver with clause learning on top of a
  SAT solver.

## Glossary

| term | meaning |
|---|---|
| assertion level | the level to which the solver jumps after a conflict; the learned clause then forces one literal |
| asserting clause | a learned clause with exactly one literal of the current level |
| backjumping | going back more than one level, past the decisions that had no part in the conflict |
| bound literal | `[x >= v]` or `[x <= v]` |
| CDCL | conflict-driven clause learning, the algorithm of modern SAT solvers |
| clause | a disjunction of literals |
| conflict | a failure found by a propagator or by a clause |
| decision level | the number of decisions on the current path; 0 is the root |
| explanation | a set of true literals that force a literal through one constraint |
| first UIP (1UIP) | the literal of the current level, closest to the conflict, through which all the paths from the decision to the conflict go |
| implication trail | the ordered list of the literals that became true, with their levels and reasons |
| LBD | literal block distance: the number of distinct levels in a clause; a low LBD predicts a useful clause |
| LCG | lazy clause generation: CDCL inside a CP solver, with clauses made from explanations during the search |
| learned clause | a clause that conflict analysis adds to the problem |
| literal | a fact about the domains that is true, false or unassigned at a node |
| naive explanation | all the current bounds of the variables of a propagator; always correct, often long |
| nogood | a conjunction of literals that must not all hold; the negation of a clause |
| phase saving | choosing the value side that a variable had the last time |
| reason | what made a literal true: a decision, a propagator, a learned clause or a root fact |
| restart | going back to the root, keeping what was learned |
| unit propagation | setting the last literal of a clause whose other literals are all false |
| VSIDS | variable state independent decaying sum: the activity-based variable choice of Chaff |
| watched literals | the two literals of a clause that the solver watches; it visits the clause only when one of them becomes false |


[cdcl-wiki]: https://en.wikipedia.org/wiki/Conflict-driven_clause_learning
[stuckey-sat2013]: https://www.cs.helsinki.fi/group/sat2013/slides/SAT2013-stuckey.pdf
[ohrimenko2007]: https://people.eng.unimelb.edu.au/pstuckey/papers/cp07a.pdf
[stuckey-ppdp2013]: https://people.eng.unimelb.edu.au/pstuckey/PPDP2013.pdf
[cdcl-handbook]: https://www.cs.princeton.edu/~zkincaid/courses/fall18/readings/SATHandbook-CDCL.pdf
[lal-2024]: https://cis.upenn.edu/~cis1921/files24/lecture10/LECTURE_10_SLIDES.pdf
[grasp]: https://doi.org/10.1109/12.769433
[grasp-free]: https://www.cs.cmu.edu/~emc/15-820A/reading/grasp_iccad96.pdf
[chaff]: https://doi.org/10.1145/378239.379017
[chaff-free]: https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf
[minisat]: https://doi.org/10.1007/978-3-540-24605-3_37
[minisat-free]: http://minisat.se/downloads/MiniSat.pdf
[ohrimenko2009]: https://doi.org/10.1007/s10601-008-9064-x
[feydy2009]: https://doi.org/10.1007/978-3-642-04244-7_29
[stuckey-cpaior2010]: https://doi.org/10.1007/978-3-642-13520-0_3
[chu-thesis]: https://hdl.handle.net/11343/36679
[lecoutre-jsat]: https://doi.org/10.3233/SAT190009
[lecoutre-ijcai]: https://www.cril.univ-artois.fr/~sais/papers/IJCAI2007NRR.pdf
[achterberg]: https://doi.org/10.1016/j.disopt.2006.10.006
[downing-alldiff]: https://research.monash.edu/en/publications/explaining-alldifferent/
[downing-flow]: https://doi.org/10.1007/978-3-642-29828-8_10
[gange-mdd]: https://doi.org/10.1007/s10601-011-9111-x
[vilim-unary]: https://vilim.eu/petr/cpaior2005.pdf
[schutt-cumulative]: https://doi.org/10.1007/s10601-010-9103-2
[schutt-ttef]: https://doi.org/10.1007/978-3-642-38171-3_16
[francis-circuit]: https://people.eng.unimelb.edu.au/pstuckey/papers/explainingcircuit.pdf
[lrb]: https://cs.uwaterloo.ca/~ppoupart/publications/sat/learning-rate-branching-heuristic-SAT.pdf
[chb]: https://cs.uwaterloo.ca/~ppoupart/publications/sat/sat-erwa.pdf
[glucose]: https://www.ijcai.org/Proceedings/09/Papers/074.pdf
[chuffed]: https://github.com/chuffed/chuffed
[cpsat-readme]: https://github.com/google/or-tools/blob/stable/ortools/sat/README.md
