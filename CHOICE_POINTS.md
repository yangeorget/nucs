# Choice points and trailing

How NuCS saves and restores search state. Companion to the *Choice points copy the whole domains array
instead of trailing* section of `ARCHITECTURE.md`, which gives the summary; this document gives the
mechanism. Code lives in `nucs/solvers/choice_points.py`, with the branching half in
`nucs/heuristics/*_dom_heuristic.py` and the search loop in `solve_one` (`nucs/solvers/backtrack_solver.py`).

## Two mechanisms, not one

NuCS uses **both** state-restoration strategies, each where it is cheaper:

| state | mechanism | why |
|-------|-----------|-----|
| domains | **copying** — full snapshot per choice point | a contiguous `int32` memcpy of `2 · domain_nb` cells; restore is a pointer decrement |
| entailment | **trailing** — depth per propagator + undo log | indexed by propagator, not variable, and monotonic within a branch, so most levels change nothing |

Nothing else is saved. The propagation queue is *not* restored on backtrack: `backtrack` reschedules only
the propagators the refuted decision affects, which is the point of recording that decision (below).

## Data structures

### The copying stacks

Parallel stacks of height `H = stks_max_height` (`BacktrackSolver.__init__`, default 8192), allocated once
and sharing a single top:

| array | shape | dtype | per-level meaning |
|-------|-------|-------|-------------------|
| `domains_stk` | `(H, domain_nb, 2)` | int32 | domains snapshot; `[MIN]`/`[MAX]` per variable |
| `domain_update_stk` | `(H, 2)` | uint32 | the level's *pending, unpropagated* decision: `[DOM_UPDATE_VARIABLE, DOM_UPDATE_EVENTS]` |
| `unbound_variable_nb_stk` | `(H,)` | uint32 | number of still-unbound variables at that level |
| `stks_top` | `(1,)` | uint32 | index of the top, shared by all three |

`stks_top` is a one-element array rather than an `int` because it is mutable state shared across separately
jitted functions — `solve_one`, the domain heuristics, `backtrack`, the consistency algorithm all write it,
and Numba has no way to pass a scalar by reference.

`cp_put` pushes `domains_stk` and `unbound_variable_nb_stk` in lockstep but keeps them separate arrays; the
comment there records why folding the count into `domains_stk` as an extra row was rejected.

### The entailment trail

| array | shape | dtype | meaning |
|-------|-------|-------|---------|
| `entailed_propagator_depths` | `(P,)` | int32 | depth at which propagator `p` was entailed, `-1` when active |
| `entailment_trail` | `(P + 1,)` | int32 | `[0]` is the trail size; `[1:size]` are propagator ids in entailment order |

`P = propagator_nb`. The size lives in cell 0 so the trail is a single self-contained array to thread
through the jitted call chain.

## What `top` means

At any moment:

- **levels `0 .. top-1`** hold *pending alternatives* — branches created but not yet explored. Each carries,
  in `domain_update_stk[d]`, the decision already written into `domains_stk[d]` but never propagated.
- **level `top`** holds the live domains. This is what the consistency algorithm reads and writes
  (`domains = domains_stk[top]` in `bc_algorithm`), and what `get_solution` returns.

So the stack depth is also the search depth, and `stks_top[0] == 0` with no alternative left is exhaustion.

`cp_init` resets the whole thing: level 0 gets the initial domains, every propagator goes back to active,
the trail empties, and `top` returns to 0.

## Branching: both branches are written, only one is propagated

The subtle part. Take `split_low_dom_heuristic`, called from `solve_one` once the variable heuristic has
picked a variable:

```python
top = stks_top[0]
cp_put(domains_stk, unbound_variable_nb_stk, top)   # copy level top up to top + 1
value = (domains_stk[top, variable, MIN] + domains_stk[top, variable, MAX]) >> 1
domains_stk[top + 1, variable, MAX] = value         # branch explored now:  [min, value]
domains_stk[top, variable, MIN] = value + 1         # branch left behind:   [value+1, max]
...
domain_update_stk[top, DOM_UPDATE_VARIABLE] = variable
domain_update_stk[top, DOM_UPDATE_EVENTS] = EVENT_MASK_MIN(_GROUND)
stks_top[0] = top + 1
return events                                        # events of the branch explored now
```

Both halves of the split are applied to the domains immediately. The heuristic returns the events of the
branch it descends into, and `solve_one` schedules those; the events of the branch left behind are parked in
`domain_update_stk[top]` for whenever the search comes back.

Note the roles: the *old* top becomes the stored alternative, and the *new* top is the branch explored now.
A level's `domain_update_stk` entry therefore describes that level's own domains — already narrowed, still
unpropagated — not the decision that led to it.

## Backtracking

`backtrack` does no domain restoration at all:

1. return `False` if `top == 0` — the search is exhausted;
2. `stks_top[0] -= 1` — the previous level's snapshot becomes live again, untouched since it was written;
3. `unwind_entailment_trail(...)` — reactivate propagators entailed below the new top;
4. `update_propagators(...)` with `domain_update_stk[top]` — schedule exactly the propagators watching the
   parked decision.

Step 4 is why the queue needs no rebuild: the only unpropagated change at this level is the one recorded
when the level was pushed.

When optimizing, steps 1-4 are a loop rather than a single pass: after step 4 the objective bound is
re-applied to the resumed level (see *Optimization*), and if it wipes the level out the search pops again.

`solve()` also calls `backtrack` after yielding each solution, which is how iteration resumes: a solution is
a fully bound level, and forcing a backtrack drops to the next pending alternative. `optimize()` does the
same, having first armed `objective` with the bound the solution establishes.

## The entailment trail in detail

Recorded in `bc_algorithm` when a propagator returns `PROP_ENTAILMENT`, and only the first time:

```python
if entailed_propagator_depths[prop_idx] == -1:
    entailed_propagator_depths[prop_idx] = top
    entailment_trail[0] += 1
    entailment_trail[entailment_trail[0]] = prop_idx
```

Two properties make the undo a single scan. Entailment is **monotonic within a branch** — a propagator
entailed at depth `d` stays entailed at every depth below — so recording the shallowest depth is enough, and
the `== -1` guard keeps it there. And the trail is **ordered by non-decreasing depth**, because entries are
appended in the order they are entailed and `top` only grows between two appends within a branch.

`unwind_entailment_trail(depths, trail, top)` therefore pops from the end while
`depths[trail[size]] > top`, resetting each to `-1`, and stops at the first entry that is still valid.

Elsewhere the flag is read as a plain `!= -1` test: `update_propagators` and `update_domains` skip entailed
propagators when scheduling, so an entailed propagator costs one comparison rather than a call.

## Optimization

`optimize` re-enters the search after each local optimum, and the two `OPTIM_*` modes differ exactly in what
they do to the choice-point stacks (`BacktrackSolver._advance_after_optimum`):

| mode | function | effect on the stacks |
|------|----------|----------------------|
| `OPTIM_RESET` | `cp_init` + `fix_choice_point` | throws the stacks away, restarts from the initial domains with the objective bound tightened at level 0 |
| `OPTIM_PRUNE` | arms `objective`, then `backtrack` | keeps the stacks; the bound is re-applied by `backtrack` to each level as it is resumed |

`OPTIM_PRUNE` rests on the observation that **the branch-and-bound bound is not backtrackable**. It holds for
the whole remaining search, so it is solver state rather than choice-point state: `_advance_after_optimum`
writes it into `objective` (`[OBJ_VARIABLE, OBJ_BOUND, OBJ_VALUE]`, with `OBJ_VARIABLE == -1` when not
optimizing) and then simply backtracks.

`backtrack` is what applies it, through `tighten_objective`, to whichever level it lands on. The tightening
is monotone and idempotent, so a level that is resumed twice is tightened once in effect, and a level the
bound wipes out can no longer hold an improving solution — `backtrack` keeps popping. That is why no level
is ever pruned in advance, and why `backtrack` is a loop rather than a single pop.

Nothing walks the stored levels. An earlier design did: it rewrote the bound into every level and dropped
one level per wipe-out, which assumes the wiped levels are the deepest ones. Two-way splits guarantee that;
`value_dom_heuristic`'s three-way split does not, and the mismatch discarded a surviving level and hung the
solver (`test_prune_terminates_on_a_three_way_split`).

## Where the counters are maintained

`unbound_variable_nb_stk[top]` is the solved test — `bc_algorithm` returns `PROBLEM_BOUND` when the queue
drains and the count is 0 — and it is maintained incrementally from three places, never recomputed:

- `update_domains`, when propagation grounds a variable (`events |= EVENT_MASK_GROUND`);
- the domain heuristics, for whichever of the two branches it grounds;
- `fix_choice_point` and `tighten_objective`, when tightening the objective bound grounds the objective.
