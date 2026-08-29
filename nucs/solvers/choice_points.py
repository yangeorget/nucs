###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024-2026 - Yan Georget
###############################################################################

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    CHOICE_POINT_BOUND,
    CHOICE_POINT_TRAIL_MARK,
    CHOICE_POINT_VALUE,
    CHOICE_POINT_VARIABLE,
    DECISION_EQ,
    DECISION_GT,
    DECISION_LE,
    EVENT_MASK_GROUND,
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    EVENT_MASK_NONE,
    MAX,
    MIN,
    OBJ_BOUND,
    OBJ_VALUE,
    OBJ_VARIABLE,
    STATS_IDX_SOLVER_BACKTRACK_NB,
)
from nucs.propagators.propagators import update_propagators


@njit(cache=True, inline="always")
def unbound_index(state: NDArray) -> int:
    """
    Returns the flat index of the unbound-variable count, which is the last cell of the state.

    It sits at the end, after the entailment flags, rather than between them and the domains, so that
    its index does not depend on the number of propagators.

    :param state: all the backtrackable state
    :type state: NDArray

    :return: the index of the count in state
    :rtype: int
    """
    return len(state) - 1


@njit(cache=True, inline="always")
def trail_push(trail_log: NDArray, trail_indices: NDArray, trail_size: int, flat: int, old: int) -> int:
    """
    Records a cell's old value unconditionally, for state that guards itself.

    An entailment flag only ever goes from active to entailed, and only where the caller has just tested
    that it was active, so it cannot be written twice in a choice point and needs no positional guard. The
    position is still recorded, so that trail_undo can clear it like any other.

    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_indices: the index of the last trail entry per cell
    :type trail_indices: NDArray
    :param trail_size: the number of entries currently on the trail
    :type trail_size: int
    :param flat: the index of the cell in state
    :type flat: int
    :param old: the value currently in that cell
    :type old: int

    :return: the new trail size
    :rtype: int
    """
    trail_log[trail_size, 0] = flat
    trail_log[trail_size, 1] = old
    trail_indices[flat] = trail_size
    return trail_size + 1


@njit(cache=True, inline="always")
def trail_set(
    state: NDArray,
    trail_log: NDArray,
    trail_indices: NDArray,
    mark: int,
    trail_size: int,
    flat: int,
    old: int,
    value: int,
) -> int:
    """
    Writes a cell of the backtrackable state, recording its old value if this choice point has not already.

    The rule a write barrier has to implement is exactly: a write to flat may skip the trail iff the trail
    already holds a live entry *for flat* at an index >= the current choice point's mark. Redundant entries are
    always safe -- undo is LIFO, so an extra entry restores a value at least as old as the one after it --
    but a missing entry corrupts.

    "Live" has to mean live *for this cell*, and a position inside the live range is not by itself
    evidence of that: popping the trail and letting it regrow leaves trail_indices[flat] pointing at an index
    some other cell has since claimed. Rather than pay to re-read the entry and check whose it is -- a random
    load on exactly the path that was about to be cheap -- trail_undo clears the position of every entry
    it pops, so a position is stale only by being -1. That moves the cost to the pop, which happens once
    per trailed write, and off the skip, which happens as often as a fixpoint re-narrows a bound.

    Stating the rule positionally, rather than approximating it with a generation counter stamped per
    cell, means there is no counter to bump -- and therefore no site that can forget to bump one. cp_init
    runs at solve time on every OPTIM_RESET, and every stack mutation is covered for free.

    :param state: all the backtrackable state
    :type state: NDArray
    The trail size is taken and returned rather than read out of its array, so that a caller writing
    several cells in a row -- which is every caller -- keeps it in a register instead of reloading it
    across each write.

    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param mark: the trail size when the current choice point branched
    :type mark: int
    :param trail_size: the number of entries currently on the trail
    :type trail_size: int
    :param flat: the index of the cell in state
    :type flat: int
    :param old: the value currently in that cell, which every caller has already had to read
    :type old: int
    :param value: the value to write
    :type value: int

    :return: the new trail size
    :rtype: int
    """
    entry = trail_indices[flat]
    if not mark <= entry < trail_size:  # no live entry for this cell at this choice point
        trail_log[trail_size, 0] = flat
        trail_log[trail_size, 1] = old
        trail_indices[flat] = trail_size
        trail_size += 1
    state[flat] = value
    return trail_size


@njit(cache=True)
def trail_undo(state: NDArray, trail_log: NDArray, trail_indices: NDArray, trail_top: NDArray, mark: int) -> None:
    """
    Restores every cell the trail recorded since a mark, and forgets where those entries were.

    One loop, no discriminator: restoring a domain bound, the unbound-variable count and an entailed
    propagator's flag are the same pair of instructions. That is what the single flat state array buys.

    Clearing the position is what lets the write barrier trust a position that is merely in range; a
    cell whose entry has been popped is left at -1, which no range test can accept. It costs one store
    per popped entry, on the path that is already touching the entry, and saves the barrier a random
    read on every write it skips.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_indices: the index of the last trail entry per cell
    :type trail_indices: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param mark: the trail size to restore to
    :type mark: int
    """
    idx = trail_top[0]
    while idx > mark:
        idx -= 1
        flat = trail_log[idx, 0]
        state[flat] = trail_log[idx, 1]
        trail_indices[flat] = -1
    trail_top[0] = mark


@njit(cache=True)
def cp_init(
    state: NDArray,
    entailed: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    choice_point_stk: NDArray,
    choice_point_top: NDArray,
    domains_arr: NDArray,
    unbound_variable_nb: int,
) -> None:
    """
    Initializes the choice points.

    trail_indices has to be cleared rather than left to invalidate itself: the guard reads trail_indices[flat] as
    a trail index, and a position left over from a previous search can fall inside the new live range and
    suppress a write that needed trailing.

    :param state: all the backtrackable state
    :type state: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param trail_indices: the index of the last trail entry per cell
    :type trail_indices: NDArray
    :param choice_point_stk: the per-choice-point metadata
    :type choice_point_stk: NDArray
    :param choice_point_top: the index of the top of the choice points as a Numpy array
    :type choice_point_top: NDArray
    :param domains_arr: the domains
    :type domains_arr: NDArray
    :param unbound_variable_nb: the number of unbound variables
    :type unbound_variable_nb: int
    """
    for variable in range(len(domains_arr)):
        state[variable << 1] = domains_arr[variable, MIN]
        state[(variable << 1) | 1] = domains_arr[variable, MAX]
    state[unbound_index(state)] = unbound_variable_nb
    entailed.fill(0)
    trail_indices.fill(-1)
    choice_point_stk[0, CHOICE_POINT_TRAIL_MARK] = 0
    trail_top[0] = choice_point_top[0] = 0


@njit(cache=True, inline="always")
def tighten(
    state: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    mark: int,
    variable: int,
    new_min: int,
    new_max: int,
) -> int:
    """
    Writes a variable's domain and returns the events the write triggers.

    This is the form to call from anywhere but the propagation loop, which uses tighten_at so as to keep
    the trail size in a register across a whole filtering rather than round-tripping it through memory
    on every bound it narrows.

    This is the only place a backtrackable domain is written. Every other kind of domain write -- a
    propagator's filtering, a branching decision, the branch-and-bound clamp, a custom consistency
    algorithm's pruning -- routes through here, so the groundness test, the unbound-variable count and
    the write barrier stay consistent by construction rather than by convention.

    Scheduling is deliberately left to the caller. The propagation loop schedules with a self-skip and a
    membership pre-test that update_propagators does not have, so folding the two loops together would
    either lose that specialization or drag six more parameters into the hottest function in the solver.

    The caller owns the wipeout test too: it is one comparison on values the caller is already holding,
    and the propagation loop can never wipe out (a propagator reports the inconsistency instead), so
    charging it for the check would be paying on the hot path for the cold callers.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param mark: the trail size when the current choice point branched
    :type mark: int
    :param variable: the variable
    :type variable: int
    :param new_min: the new min of the domain
    :type new_min: int
    :param new_max: the new max of the domain
    :type new_max: int

    :return: the events triggered by the write, EVENT_MASK_NONE when it changed nothing
    :rtype: int
    """
    events, trail_size = tighten_at(state, trail_log, trail_indices, mark, trail_top[0], variable, new_min, new_max)
    trail_top[0] = trail_size
    return events


@njit(cache=True, inline="always")
def tighten_at(
    state: NDArray,
    trail_log: NDArray,
    trail_indices: NDArray,
    mark: int,
    trail_size: int,
    variable: int,
    new_min: int,
    new_max: int,
) -> tuple[int, int]:
    """
    Writes a variable's domain, taking and returning the trail size rather than reading it out of memory.

    A filtering narrows several bounds in a row, and the propagation loop calls this for each variable of
    each propagator -- the hottest loop in the solver. Threading the trail size through keeps it in a
    register for the whole filtering: without that, every narrowed bound loads and stores the same cell,
    and each store is a memory dependency the next iteration has to wait on.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param mark: the trail size when the current choice point branched
    :type mark: int
    :param trail_size: the number of entries currently on the trail
    :type trail_size: int
    :param variable: the variable
    :type variable: int
    :param new_min: the new min of the domain
    :type new_min: int
    :param new_max: the new max of the domain
    :type new_max: int

    :return: the events triggered by the write and the new trail size
    :rtype: Tuple[int, int]
    """
    # the barrier is per bound, not per variable: a filtering writes both MIN and MAX within one choice point,
    # so a guard indexed by variable would suppress the second write and never restore MAX
    flat = variable << 1
    old_min = state[flat]
    old_max = state[flat | 1]
    events = EVENT_MASK_NONE
    if old_min != new_min:
        trail_size = trail_set(state, trail_log, trail_indices, mark, trail_size, flat, old_min, new_min)
        events = EVENT_MASK_MIN
    if old_max != new_max:
        trail_size = trail_set(state, trail_log, trail_indices, mark, trail_size, flat | 1, old_max, new_max)
        events |= EVENT_MASK_MAX
    if events != EVENT_MASK_NONE and old_min != old_max and new_min == new_max:
        events |= EVENT_MASK_GROUND
        unbound = unbound_index(state)
        old_unbound = state[unbound]
        trail_size = trail_set(state, trail_log, trail_indices, mark, trail_size, unbound, old_unbound, old_unbound - 1)
    return events, trail_size


@njit(cache=True, inline="always")
def park(choice_point_stk: NDArray, choice_point: int, mark: int, variable: int, bound: int, value: int) -> None:
    """
    Records on a choice point the alternative to apply when the search resumes it.

    :param choice_point_stk: the per-choice-point metadata
    :type choice_point_stk: NDArray
    :param choice_point: the choice point
    :type choice_point: int
    :param mark: the trail size when the choice point branched
    :type mark: int
    :param variable: the variable of the alternative
    :type variable: int
    :param bound: the side of its domain the alternative tightens
    :type bound: int
    :param value: the value the alternative tightens that side to
    :type value: int
    """
    choice_point_stk[choice_point, CHOICE_POINT_TRAIL_MARK] = mark
    choice_point_stk[choice_point, CHOICE_POINT_VARIABLE] = variable
    choice_point_stk[choice_point, CHOICE_POINT_BOUND] = bound
    choice_point_stk[choice_point, CHOICE_POINT_VALUE] = value


@njit(cache=True)
def branch(
    state: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    choice_point_stk: NDArray,
    choice_point_top: NDArray,
    variable: int,
    kind: int,
    value: int,
) -> int:
    """
    Applies a decision: explores one branch and parks the alternatives for the search to resume.

    A push copies nothing. It records the trail position -- so that returning to this choice point is a matter of
    replaying the undo log back to it -- and the alternative to apply on arrival, which is always a
    single-bound tightening, hence monotone and idempotent, hence safe to re-apply.

    Only the explored branch is written. The alternatives stay parked, which is the point: the domains of
    a branch the search has not taken yet do not exist anywhere.

    The parked alternatives are recorded on the choice points below the explored one, deepest first, so that
    backtracking meets them in that order. DECISION_EQ is the ternary case and parks two of them.

    The one above the deepest parked alternative gets the same mark, because it is where the search is about
    to work and every write it makes has to be undone back to here.

    An EQ value outside the domain is clamped into it rather than applied as written. min_cost returns -1
    when no value in the domain has a positive cost; clamping keeps the split a partition of the domain,
    which is what makes the enumeration complete.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param choice_point_stk: the per-choice-point metadata
    :type choice_point_stk: NDArray
    :param choice_point_top: the index of the top of the choice points as a Numpy array
    :type choice_point_top: NDArray
    :param variable: the variable being branched on
    :type variable: int
    :param kind: the kind of split, one of DECISION_LE, DECISION_GT and DECISION_EQ
    :type kind: int
    :param value: the value the domain is split at
    :type value: int

    :return: the events triggered in the explored branch
    :rtype: int
    """
    choice_point = choice_point_top[0]
    flat = variable << 1
    domain_min = state[flat]
    domain_max = state[flat | 1]
    mark = trail_top[0]
    if kind == DECISION_EQ:
        value = min(max(value, domain_min), domain_max)
        # an EQ at either end of the domain has an empty alternative on that side, so it is a two-way
        # split: normalizing here is what keeps the ternary case to genuinely ternary splits
        if value == domain_min:
            kind = DECISION_LE
        elif value == domain_max:
            kind = DECISION_GT
            value = domain_max - 1
    if kind == DECISION_LE:
        park(choice_point_stk, choice_point, mark, variable, MIN, value + 1)
        choice_point_stk[choice_point + 1, CHOICE_POINT_TRAIL_MARK] = mark
        choice_point_top[0] = choice_point + 1
        return tighten(state, trail_log, trail_top, trail_indices, mark, variable, domain_min, value)
    if kind == DECISION_GT:
        park(choice_point_stk, choice_point, mark, variable, MAX, value)
        choice_point_stk[choice_point + 1, CHOICE_POINT_TRAIL_MARK] = mark
        choice_point_top[0] = choice_point + 1
        return tighten(state, trail_log, trail_top, trail_indices, mark, variable, value + 1, domain_max)
    park(choice_point_stk, choice_point, mark, variable, MIN, value + 1)  # the shallower alternative is resumed last
    park(choice_point_stk, choice_point + 1, mark, variable, MAX, value - 1)
    choice_point_stk[choice_point + 2, CHOICE_POINT_TRAIL_MARK] = mark
    choice_point_top[0] = choice_point + 2
    return tighten(state, trail_log, trail_top, trail_indices, mark, variable, value, value)


@njit(cache=True, inline="always")
def tighten_objective(
    state: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    mark: int,
    objective: NDArray,
) -> int:
    """
    Applies the current branch-and-bound bound to the choice point the search has just resumed.

    The bound is not backtrackable: it holds for the whole remaining search, so it is re-applied to each
    choice point as the search resumes it rather than written into them all when it is found. The tightening
    is monotone and idempotent, so re-applying it to an already-tightened choice point is a no-op.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param mark: the trail size when the resumed choice point branched
    :type mark: int
    :param objective: the objective as a Numpy array of variable, bound and value
    :type objective: NDArray

    :return: the events triggered by the tightening, EVENT_MASK_NONE when it changed nothing,
             -1 when the choice point wipes out
    :rtype: int
    """
    variable = objective[OBJ_VARIABLE]
    value = objective[OBJ_VALUE]
    flat = variable << 1
    if objective[OBJ_BOUND] == MIN:
        new_min = max(value + 1, state[flat])
        new_max = state[flat | 1]
    else:
        new_min = state[flat]
        new_max = min(value - 1, state[flat | 1])
    if new_min > new_max:
        return -1
    return tighten(state, trail_log, trail_top, trail_indices, mark, variable, new_min, new_max)


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    state: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    choice_point_stk: NDArray,
    choice_point_top: NDArray,
    entailed: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    priorities: NDArray,
    propagator_nb: int,
    objective: NDArray,
) -> bool:
    """
    Backtracks to the deepest choice point that can still hold a solution.

    Popping a choice point is replaying the undo log back to its mark, then applying the alternative it
    parked -- through the same write barrier as any other write, so that the refutation is itself
    undone when the search later backtracks past this choice point.

    The undo reactivates the propagators entailed below this choice point as it goes: an entailment flag is a
    cell of the same state array as a domain bound, so there is nothing separate left to unwind.

    When optimizing, the objective bound is re-applied to the choice point being resumed; one it wipes out
    can no longer hold an improving solution, so the search keeps popping. This is why none of them ever has
    to be pruned in advance.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param choice_point_stk: the per-choice-point metadata
    :type choice_point_stk: NDArray
    :param choice_point_top: the index of the top of the choice points as a Numpy array
    :type choice_point_top: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :type triggered_propagators: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
    :param propagator_nb: the number of propagators
    :type propagator_nb: int
    :param objective: the objective as a Numpy array of variable, bound and value,
                      whose variable is -1 when not optimizing
    :type objective: NDArray

    :return: true iff it is possible to backtrack
    :rtype: bool
    """
    optimized_variable = objective[OBJ_VARIABLE]
    while choice_point_top[0] > 0:
        choice_point_top[0] -= 1
        choice_point = choice_point_top[0]
        statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
        mark = choice_point_stk[choice_point, CHOICE_POINT_TRAIL_MARK]
        trail_undo(state, trail_log, trail_indices, trail_top, mark)
        variable = choice_point_stk[choice_point, CHOICE_POINT_VARIABLE]
        flat = variable << 1
        if choice_point_stk[choice_point, CHOICE_POINT_BOUND] == MIN:
            new_min = max(choice_point_stk[choice_point, CHOICE_POINT_VALUE], state[flat])
            new_max = state[flat | 1]
        else:
            new_min = state[flat]
            new_max = min(choice_point_stk[choice_point, CHOICE_POINT_VALUE], state[flat | 1])
        events = tighten(state, trail_log, trail_top, trail_indices, mark, variable, new_min, new_max)
        if events != EVENT_MASK_NONE:
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                priorities,
                propagator_nb,
                variable,
                events,
            )
        if optimized_variable < 0:
            return True
        events = tighten_objective(state, trail_log, trail_top, trail_indices, mark, objective)
        if events < 0:  # the choice point cannot hold an improving solution, keep popping
            continue
        if events != EVENT_MASK_NONE:
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                priorities,
                propagator_nb,
                optimized_variable,
                events,
            )
        return True
    return False


@njit(cache=True)
def fix_choice_point(
    state: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    variable: int,
    value: int,
    bound: int,
) -> bool:
    """
    Fixes the domain of the variable being optimized at the root, after a reset.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (flat index, old value) pairs
    :type trail_log: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param variable:  the variable being optimized
    :type variable: int
    :param value: the current optimal value for the variable
    :type value: int
    :param bound: the bound being optimized
    :type bound: int

    :return: true iff the resulting domain is non-empty
    :rtype: bool
    """
    flat = variable << 1
    if bound == MIN:
        new_min = value + 1
        new_max = state[flat | 1]
    else:
        new_min = state[flat]
        new_max = value - 1
    if new_min > new_max:
        return False
    tighten(state, trail_log, trail_top, trail_indices, 0, variable, new_min, new_max)
    return True
