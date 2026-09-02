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
    DECISION_EQ,
    DECISION_GT,
    DECISION_LE,
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_NONE,
    OBJECTIVE_BOUND,
    OBJECTIVE_VALUE,
    OBJECTIVE_VARIABLE,
)
from nucs.propagators.propagators import update_propagators
from nucs.solvers.state import tighten, trail_undo, unbound_index
from nucs.statistics import STATS_IDX_SOLVER_BACKTRACK_NB

# Choice point metadata columns.
# One row per choice point, holding what *describes* it rather than what it changed: the trail
# position at its decision point, and the single-bound tightening to apply when the search resumes it.
# None of it is trailed -- trailing the decision would erase the very thing backtrack is about to apply.
CHOICE_POINT_TRAIL_MARK = 0  # the trail size when the choice point branched, the point trail_undo restores to
CHOICE_POINT_VARIABLE = 1  # the variable of the parked alternative
CHOICE_POINT_BOUND = 2  # the side of its domain the alternative tightens
CHOICE_POINT_VALUE = 3  # the value the alternative tightens that side to
CHOICE_POINT_WIDTH = 4  # the number of columns of a choice point


@njit(cache=True)
def choice_point_init(
    state: NDArray,
    entailed: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    choice_point_stk: NDArray,
    choice_point_top: NDArray,
    domains: NDArray,
    unbound_variable_nb: int,
) -> None:
    """
    Initializes the choice points.

    trail_indices has to be cleared rather than left to invalidate itself: the guard reads trail_indices[cell_idx] as
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
    :param domains: the domains
    :type domains: NDArray
    :param unbound_variable_nb: the number of unbound variables
    :type unbound_variable_nb: int
    """
    for variable in range(len(domains)):
        cell_idx = variable << 1
        state[cell_idx] = domains[variable, DOMAIN_MIN]
        state[cell_idx | 1] = domains[variable, DOMAIN_MAX]
    state[unbound_index(state)] = unbound_variable_nb
    entailed.fill(0)
    trail_indices.fill(-1)
    choice_point_stk[0, CHOICE_POINT_TRAIL_MARK] = trail_top[0] = choice_point_top[0] = 0


@njit(cache=True, inline="always")
def park_alternative(
    choice_point_stk: NDArray, choice_point: int, mark: int, variable: int, bound: int, value: int
) -> None:
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
    :param trail_log: the undo log of (cell index, old value) pairs
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
    cell_idx = variable << 1
    domain_min = state[cell_idx]
    domain_max = state[cell_idx | 1]
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
        park_alternative(choice_point_stk, choice_point, mark, variable, DOMAIN_MIN, value + 1)
        choice_point_stk[choice_point + 1, CHOICE_POINT_TRAIL_MARK] = mark
        choice_point_top[0] = choice_point + 1
        return tighten(state, trail_log, trail_top, trail_indices, mark, variable, domain_min, value)
    if kind == DECISION_GT:
        park_alternative(choice_point_stk, choice_point, mark, variable, DOMAIN_MAX, value)
        choice_point_stk[choice_point + 1, CHOICE_POINT_TRAIL_MARK] = mark
        choice_point_top[0] = choice_point + 1
        return tighten(state, trail_log, trail_top, trail_indices, mark, variable, value + 1, domain_max)
    # the shallower alternative is resumed last
    park_alternative(choice_point_stk, choice_point, mark, variable, DOMAIN_MIN, value + 1)
    park_alternative(choice_point_stk, choice_point + 1, mark, variable, DOMAIN_MAX, value - 1)
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
    :param trail_log: the undo log of (cell index, old value) pairs
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
    variable = objective[OBJECTIVE_VARIABLE]
    value = objective[OBJECTIVE_VALUE]
    cell_idx = variable << 1
    if objective[OBJECTIVE_BOUND] == DOMAIN_MIN:
        new_min = max(value + 1, state[cell_idx])
        new_max = state[cell_idx | 1]
    else:
        new_min = state[cell_idx]
        new_max = min(value - 1, state[cell_idx | 1])
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
    objective: NDArray,
) -> bool:
    """
    Backtracks to the deepest choice point that can still hold a solution.

    Popping a choice point is replaying the undo log back to its mark, then applying the alternative it
    parked -- through the same write barrier as any other write, so that the alternative is itself
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
    :param trail_log: the undo log of (cell index, old value) pairs
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
    :param objective: the objective as a Numpy array of variable, bound and value,
                      whose variable is -1 when not optimizing
    :type objective: NDArray

    :return: true iff it is possible to backtrack
    :rtype: bool
    """
    optimized_variable = objective[OBJECTIVE_VARIABLE]
    while choice_point_top[0] > 0:
        choice_point_top[0] -= 1
        choice_point = choice_point_top[0]
        statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
        mark = choice_point_stk[choice_point, CHOICE_POINT_TRAIL_MARK]
        trail_undo(state, trail_log, trail_indices, trail_top, mark)
        variable = choice_point_stk[choice_point, CHOICE_POINT_VARIABLE]
        cell_idx = variable << 1
        if choice_point_stk[choice_point, CHOICE_POINT_BOUND] == DOMAIN_MIN:
            new_min = max(choice_point_stk[choice_point, CHOICE_POINT_VALUE], state[cell_idx])
            new_max = state[cell_idx | 1]
        else:
            new_min = state[cell_idx]
            new_max = min(choice_point_stk[choice_point, CHOICE_POINT_VALUE], state[cell_idx | 1])
        events = tighten(state, trail_log, trail_top, trail_indices, mark, variable, new_min, new_max)
        if events != EVENT_MASK_NONE:
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                priorities,
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
                optimized_variable,
                events,
            )
        return True
    return False


@njit(cache=True)
def tighten_objective_at_root(
    state: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    variable: int,
    value: int,
    bound: int,
) -> bool:
    """
    Applies the branch-and-bound bound to the root, after a reset.

    The OPTIM_RESET half of what tighten_objective does for OPTIM_PRUNE. It needs no clamp against the
    domain it writes: the domains are the initial ones, so the bound is a tightening by construction.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (cell index, old value) pairs
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
    cell_idx = variable << 1
    if bound == DOMAIN_MIN:
        new_min = value + 1
        new_max = state[cell_idx | 1]
    else:
        new_min = state[cell_idx]
        new_max = value - 1
    if new_min > new_max:
        return False
    tighten(state, trail_log, trail_top, trail_indices, 0, variable, new_min, new_max)
    return True
