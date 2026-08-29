# Choice points and trailing

How NuCS saves and restores search state. Companion to the *Backtrackable state is trailed, not copied*
section of `ARCHITECTURE.md`, which gives the summary; this document gives the mechanism. Code lives in
`nucs/solvers/choice_points.py`, with the search loop in `solve_one`
(`nucs/solvers/backtrack_solver.py`).

## One mechanism

Everything backtrackable lives in one flat `int32` array and is restored by one undo log:

```
              0                     2n                2n+P      2n+P+1
state (int32) [ ----- domains ----- | --- entailed --- | unbound ]     n = domain_nb, P = propagator_nb

   domains  = state[:2n].reshape(domain_nb, 2)      an int32[:, ::1] view of the same memory
   entailed = state[2n : 2n+P]                      1 when a propagator is entailed
   unbound  = state[-1]                             the number of still-unbound variables

   flat index of (variable, bound) = (variable << 1) | bound
   flat index of propagator p      = 2n + p
```

A trail entry is `(flat index, old value)` with no discriminator, so **one** `trail_undo` loop restores a
domain bound, reactivates an entailed propagator and rolls back the unbound count with the same
instruction. That is the point of the single array.

The count sits at the end rather than between the two regions so that its index does not depend on the
number of propagators.

Nothing else is saved. The propagation queue is *not* restored on backtrack: `backtrack` reschedules only
the propagators the refuted decision affects, which is the point of recording that decision (below).

### Trailed, versus per-choice-point

The line is not "O(changes) versus O(1)". It is:

- **Backtrackable state** — a value that existed before the choice point and must come back afterwards. Trail it.
  Domain bounds, entailment flags, the unbound count.
- **Per-choice-point metadata** — a value that *describes* the choice point and has no meaning outside it. Index it by
  `choice_point_top` and let the pointer decrement restore it.

The decision columns of `choice_point_stk` prove the line is real rather than a preference: trailing them would
be *wrong*, not merely slower, because undoing to the choice point's own mark would erase the very decision
`backtrack` is about to apply.

## Data structures

### Trailed

| array | shape | dtype | role |
|-------|-------|-------|------|
| `state` | `(2·domain_nb + propagator_nb + 1,)` | int32 | all the backtrackable state; `domains` and `entailed` are views into it |
| `trail_log` | `(T, 2)` | int32 | the undo log: `[flat index, old value]` |
| `trail_top` | `(1,)` | int32 | the trail size |
| `trail_indices` | `(len(state),)` | int32 | index of the last trail entry per cell, `-1` when none |

### Per-choice point

| array | shape | dtype | role |
|-------|-------|-------|------|
| `choice_point_stk` | `(H, 4)` | int32 | `[CHOICE_POINT_TRAIL_MARK, CHOICE_POINT_VARIABLE, CHOICE_POINT_BOUND, CHOICE_POINT_VALUE]` |
| `choice_point_top` | `(1,)` | uint32 | the index of the top |

`choice_point_stk[d]` holds the trail position at which choice point `d` branched, and the alternative to apply when the
search comes back to it. The four columns share an index *and* a schedule — branching writes all four,
`backtrack` reads all four — so at 16 bytes a choice point's whole metadata is one aligned chunk, touched as a
unit. (Contrast the per-propagator arrays, which share an index but are read at wildly different
frequencies; merging *those* measured worse, and `ARCHITECTURE.md` records it.)

`choice_point_top` and `trail_top` are one-element arrays rather than `int`s because they are mutable state shared
across separately jitted functions, and Numba has no way to pass a scalar by reference.

`H` and `T` are starting sizes, not ceilings: see *Growing* below.

## The write barrier

Every backtrackable write goes through `trail_set`, and the rule it implements is exactly:

> a write to `flat` may skip the trail **iff** the trail already holds a live entry *for `flat`* at an
> index `>= choice_point_stk[choice_point_top[0], CHOICE_POINT_TRAIL_MARK]`.

```python
entry = trail_indices[flat]
if not mark <= entry < trail_size:
    trail_log[trail_size, 0] = flat
    trail_log[trail_size, 1] = old
    trail_indices[flat] = trail_size
    trail_size += 1
state[flat] = value
```

Redundant entries are always safe — undo is LIFO, so an extra entry restores a value at least as old as
the one after it — but a missing entry corrupts.

**"For `flat`" is load-bearing, and a position inside the live range does not establish it.** Popping the
trail and letting it regrow leaves `trail_indices[flat]` pointing at an index another cell has since
claimed. So
`trail_undo` clears the position of every entry it pops, and a position is stale only by being `-1`. The
cost lands on the pop, once per trailed write, rather than on the skip, which happens as often as a
fixpoint re-narrows a bound.

**The barrier is per bound, not per variable.** `update_domains` writes `MIN` and `MAX` in the same choice point,
so a guard indexed by variable would suppress the second write and never restore `MAX`.

Two things follow from stating the rule positionally rather than with a generation counter stamped per
cell. There is no counter to bump, so no site — `choice_point_init`, which runs at *solve* time on every
`OPTIM_RESET`, `backtrack`, or any future stack mutation — can forget to bump one. And there is no
overflow: a monotonic world id would wrap `int32` after ~1e9 nodes, which NuCS reaches.

Entailment is the one exception, and takes `trail_push` instead: a flag only ever goes from active to
entailed, and it is only written where the caller has just read it as clear, so it cannot be written twice
in a choice point and needs no positional guard. Its position is still recorded, so `trail_undo` can clear it like
any other.

`tighten` is the only place a domain is written — a propagator's filtering, a branching decision, the
branch-and-bound clamp, a custom consistency algorithm's pruning — so the groundness test, the unbound
count and the barrier stay consistent by construction. Scheduling is deliberately *not* part of it: the
propagation loop schedules with a self-skip and a membership pre-test that `update_propagators` does not
have. The propagation loop calls `tighten_at`, which takes and returns the trail size rather than reading
it out of memory, so a filtering keeps it in a register across every bound it narrows.

## What `choice_point_top` means

`choice_point_top[0]` is the search depth. `state` always holds the domains of the choice point the search is *at*; the
choice points below it hold, in `choice_point_stk`, the trail position to rewind to and the alternative to apply.
`choice_point_top[0] == 0` with no alternative left is exhaustion.

`choice_point_init` resets the whole thing: the initial domains, every propagator active, the trail empty, every
position `-1`, and `choice_point_top` back to 0.

## Branching: only the explored branch is written

`solve_one` asks the domain heuristic *where* to split — a kind and a value, nothing more — and `branch`
does the rest:

```python
mark = trail_top[0]
park(choice_point_stk, choice_point, mark, variable, MIN, value + 1)      # the alternative, for later
choice_point_stk[choice_point + 1, CHOICE_POINT_TRAIL_MARK] = mark       # the one about to be worked at
choice_point_top[0] = choice_point + 1
return tighten(..., mark, variable, domain_min, value)                   # the branch explored now
```

A push copies nothing. The alternatives are not materialised anywhere until the search reaches them —
the domains of a branch not yet taken do not exist. Every alternative is a single-bound tightening,
hence monotone and idempotent, hence safe to re-apply.

Three kinds cover all eight in-tree heuristics:

| kind | explored now | parked (deepest first) | heuristics |
|------|--------------|------------------------|------------|
| `LE` at `v` | `[min, v]` | `[v+1, max]` | `split_low` (`v` = mid), `min_value` (`v` = min) |
| `GT` at `v` | `[v+1, max]` | `[min, v]` | `split_high` (`v` = mid), `max_value` (`v` = max-1) |
| `EQ` at `v` | `[v, v]` | `[min, v-1]` then `[v+1, max]` | `mid_value`, `min_cost` |

`EQ` is the ternary case and parks two choice points. The parked order is what an enumeration sees
(`tests/heuristics/test_mid_value_dom_heuristic.py` pins it). `branch` normalizes an `EQ` at either end of
the domain to a two-way split, and clamps an `EQ` value outside the domain into it — `min_cost` returns
`-1` when no value has a positive cost, and a split has to partition the domain for the enumeration to
stay complete.

## Backtracking

1. return `False` if `top == 0` — the search is exhausted;
2. `choice_point_top[0] -= 1`;
3. `trail_undo(...)` back to `choice_point_stk[top, CHOICE_POINT_TRAIL_MARK]` — this restores the domains *and*
   reactivates the propagators entailed below this choice point, in one loop;
4. apply `choice_point_stk[top]`'s alternative through `tighten` — so the refutation is itself undone when the
   search later backtracks past this choice point — and `update_propagators` for the events it raises.

Step 4 is why the queue needs no rebuild: the only unpropagated change at this choice point is the one just
applied.

When optimizing, steps 1-4 are a loop: after step 4 the objective bound is re-applied to the resumed
choice point (see *Optimization*), and if that wipes it out the search pops again.

`solve()` also calls `backtrack` after yielding each solution, which is how iteration resumes: a solution
is a fully bound choice point, and forcing a backtrack drops to the next pending alternative. `optimize()` does
the same, having first armed `objective` with the bound the solution establishes.

## Optimization

`optimize` re-enters the search after each local optimum, and the two `OPTIM_*` modes differ exactly in
what they do to the choice points (`BacktrackSolver._advance_after_optimum`):

| mode | function | effect |
|------|----------|--------|
| `OPTIM_RESET` | `choice_point_init` + `fix_choice_point` | throws the choice points away, restarts from the initial domains with the objective bound tightened at the root |
| `OPTIM_PRUNE` | arms `objective`, then `backtrack` | keeps the choice points; the bound is re-applied by `backtrack` to each one as it is resumed |

`OPTIM_PRUNE` rests on the observation that **the branch-and-bound bound is not backtrackable**. It holds
for the whole remaining search, so it is solver state rather than choice-point state:
`_advance_after_optimum` writes it into `objective` (`[OBJ_VARIABLE, OBJ_BOUND, OBJ_VALUE]`, with
`OBJ_VARIABLE == -1` when not optimizing) and then simply backtracks.

`backtrack` is what applies it, through `tighten_objective`, to whichever choice point it lands on. The tightening
is monotone and idempotent, so a choice point resumed twice is tightened once in effect, and one the
bound wipes out can no longer hold an improving solution — `backtrack` keeps popping. That is why no choice point
is ever pruned in advance, and why `backtrack` is a loop rather than a single pop.

Nothing walks the stored choice points. An earlier design did: it rewrote the bound into every one and dropped
one per wipe-out, which assumes the wiped ones are the deepest. Two-way splits guarantee
that; a three-way split does not, and the mismatch discarded a surviving choice point and hung the solver
(`test_prune_terminates_on_a_three_way_split`).

## Growing

`trail_log` and `choice_point_stk` are caller-allocated, so they cannot grow inside `@njit`. Sizing them for their
worst case — depth × (2·`domain_nb` + 1) trail entries — would hand back the memory this representation
wins, and a hard failure would end a long optimization run for nothing. So `solve_one` checks for room
before each step, stops with `SOLVER_TRAIL_FULL` or `SOLVER_CHOICE_POINTS_FULL`, and `BacktrackSolver._grow`
doubles the array and resumes.

Nothing of the search is lost across the reallocation: `state`, the marks and the positions all still
address the same entries.

## Where the counters are maintained

`state[-1]` is the solved test — `bc_algorithm` returns `PROBLEM_BOUND` when the queue drains and the
count is 0 — and it is maintained incrementally by `tighten`, never recomputed. Since `tighten` is the
only domain write site, propagation, branching, the objective clamp and a custom consistency algorithm
all maintain it by construction.

As a trailed `int32` it can also go negative and be caught. It used to be `uint32`, where a double
decrement wrapped to 4294967295 and failed silently.
