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
    BB_BOUND,
    BB_VALUE,
    BB_VARIABLE,
    DECISION_EQ,
    DECISION_GT,
    DECISION_LE,
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_NONE,
)
from nucs.problems.problem import OFFSETS_STATE, OFFSETS_STATE_HINT
from nucs.solvers.propagation_steps import update_propagators
from nucs.solvers.state import tighten_at, unbound_index, utrail_undo
from nucs.statistics import STATS_IDX_SOLVER_BACKTRACK_NB

# Choice point metadata columns.
# One row per choice point, holding what *describes* it rather than what it changed: the positions of the two
# trails at its decision point, and the single-bound tightening to apply when the search resumes it.
# None of it is trailed -- trailing the decision would erase the very thing backtrack is about to apply.
CHOICEPOINT_UTRAIL_MARK = 0  # the undo trail size when the choice point branched, the point utrail_undo restores to
CHOICEPOINT_VARIABLE = 1  # the variable of the parked alternative
CHOICEPOINT_BOUND = 2  # the side of its domain the alternative tightens
CHOICEPOINT_VALUE = 3  # the value the alternative tightens that side to
# the implication trail size when the choice point branched, the point itrail_pop restores to: only LcgSolver writes it
CHOICEPOINT_ITRAIL_MARK = 4
CHOICEPOINT_WIDTH = 5  # the number of columns of a choice point


@njit(cache=True)
def choicepoint_init(
    state: NDArray,
    entailed: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
    domains: NDArray,
    unbound_variable_nb: int,
    offsets: NDArray,
) -> None:
    """
    Initializes the choice points.

    utrail_indices has to be cleared rather than left to invalidate itself: the guard reads utrail_indices[cell_idx] as
    an undo trail index, and a position left over from a previous search can fall inside the new live range and
    suppress a write that needed trailing.

    The trailed prefix of every propagator's state block is cleared for the same reason the domains are
    rewritten: this drops the undo trail, so whatever those cells hold is about to become unrecoverable, and what
    they hold is the invariant of the node the previous search stopped at, not of the root. Zero is what a
    block is worth at the root -- it is what the solver allocates and what every propagator reads as cold --
    so clearing is the whole restore. The untrailed hint suffixes are deliberately left alone: a hint is
    valid from any node by contract, and clearing them would throw away the warm sort permutations
    alldifferent and gcc have built up, which is the one thing an optimization restart should keep.

    The prefixes are cleared per propagator rather than as one range because a block is laid out
    [trailed | hint] and the blocks are contiguous, so the trailed cells are interleaved with the hint
    cells they must not touch. This runs once per restart, off any hot path.

    :param state: all the backtrackable state
    :type state: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per cell
    :type utrail_indices: NDArray
    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray
    :param domains: the domains
    :type domains: NDArray
    :param unbound_variable_nb: the number of unbound variables
    :type unbound_variable_nb: int
    :param offsets: the CSR offsets, whose OFFSETS_STATE and OFFSETS_STATE_HINT columns delimit the
                    trailed prefix of each propagator's state block
    :type offsets: NDArray
    """
    for variable in range(len(domains)):
        cell_idx = variable << 1
        state[cell_idx] = domains[variable, DOMAIN_MIN]
        state[cell_idx | 1] = domains[variable, DOMAIN_MAX]
    state[unbound_index(state)] = unbound_variable_nb
    for propagator in range(len(offsets) - 1):
        state[offsets[propagator, OFFSETS_STATE] : offsets[propagator, OFFSETS_STATE_HINT]] = 0
    entailed.fill(0)
    utrail_indices.fill(-1)
    choicepoints[0, CHOICEPOINT_UTRAIL_MARK] = utrail_top[0] = choicepoint_top[0] = 0


@njit(cache=True, inline="always")
def park_alternative(choicepoints: NDArray, choicepoint: int, mark: int, variable: int, bound: int, value: int) -> None:
    """
    Records on a choice point the alternative to apply when the search resumes it.

    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint: the choice point
    :type choicepoint: int
    :param mark: the undo trail size when the choice point branched
    :type mark: int
    :param variable: the variable of the alternative
    :type variable: int
    :param bound: the side of its domain the alternative tightens
    :type bound: int
    :param value: the value the alternative tightens that side to
    :type value: int
    """
    choicepoints[choicepoint, CHOICEPOINT_UTRAIL_MARK] = mark
    choicepoints[choicepoint, CHOICEPOINT_VARIABLE] = variable
    choicepoints[choicepoint, CHOICEPOINT_BOUND] = bound
    choicepoints[choicepoint, CHOICEPOINT_VALUE] = value


@njit(cache=True, inline="always")
def bounds_after_tightening(domain_min: int, domain_max: int, bound: int, value: int) -> tuple[int, int]:
    """
    Returns a domain once one of its bounds is tightened to a value.

    A value that does not tighten the bound leaves it as it is. The domain that this returns can be empty: the
    caller decides what an empty domain means.

    :param domain_min: the min of the domain
    :type domain_min: int
    :param domain_max: the max of the domain
    :type domain_max: int
    :param bound: the side of the domain to tighten, DOMAIN_MIN or DOMAIN_MAX
    :type bound: int
    :param value: the value to tighten that side to
    :type value: int

    :return: the new min and the new max of the domain
    :rtype: Tuple[int, int]
    """
    if bound == DOMAIN_MIN:
        return max(value, domain_min), domain_max
    return domain_min, min(value, domain_max)


@njit(cache=True)
def branch(
    state: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
    variable: int,
    kind: int,
    value: int,
) -> int:
    """
    Applies a decision: explores one branch and parks the alternatives for the search to resume.

    A push copies nothing. It records the undo trail position -- so that returning to this choice point is a matter of
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
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray
    :param variable: the variable being branched on
    :type variable: int
    :param kind: the kind of split, one of DECISION_LE, DECISION_GT and DECISION_EQ
    :type kind: int
    :param value: the value the domain is split at
    :type value: int

    :return: the events triggered in the explored branch
    :rtype: int
    """
    choicepoint = choicepoint_top[0]
    cell_idx = variable << 1
    domain_min = state[cell_idx]
    domain_max = state[cell_idx | 1]
    mark = utrail_top[0]
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
        park_alternative(choicepoints, choicepoint, mark, variable, DOMAIN_MIN, value + 1)
        choicepoints[choicepoint + 1, CHOICEPOINT_UTRAIL_MARK] = mark
        choicepoint_top[0] = choicepoint + 1
        new_min, new_max = domain_min, value
    elif kind == DECISION_GT:
        park_alternative(choicepoints, choicepoint, mark, variable, DOMAIN_MAX, value)
        choicepoints[choicepoint + 1, CHOICEPOINT_UTRAIL_MARK] = mark
        choicepoint_top[0] = choicepoint + 1
        new_min, new_max = value + 1, domain_max
    else:
        # the shallower alternative is resumed last
        park_alternative(choicepoints, choicepoint, mark, variable, DOMAIN_MIN, value + 1)
        park_alternative(choicepoints, choicepoint + 1, mark, variable, DOMAIN_MAX, value - 1)
        choicepoints[choicepoint + 2, CHOICEPOINT_UTRAIL_MARK] = mark
        choicepoint_top[0] = choicepoint + 2
        new_min, new_max = value, value
    # the undo trail size is the mark: the choice point has just been pushed at it
    events, utrail_size = tighten_at(state, utrail_log, utrail_indices, mark, mark, variable, new_min, new_max)
    utrail_top[0] = utrail_size
    return events


@njit(cache=True, inline="always")
def tighten_objective(
    state: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    mark: int,
    variable: int,
    value: int,
    bound: int,
) -> int:
    """
    Applies a branch-and-bound bound to one choice point, wiping it out when nothing improving is left.

    The bound is not backtrackable: it holds for the whole remaining search, so backtrack re-applies it to
    each choice point as the search resumes it rather than writing it into them all when it is found. The
    tightening is monotone and idempotent, so re-applying it to an already-tightened choice point is a no-op.

    OPTIM_RESET calls this at the root instead, with a mark of 0, once per improving solution. The clamp
    against the domain is redundant there -- the domains are the initial ones and value is a value some
    solution reached, so it lies inside them -- but it costs one comparison on a path that runs once per
    solution, and it is what keeps this function safe against its callers rather than dependent on them.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the choice point being tightened branched, 0 at the root
    :type mark: int
    :param variable: the variable being optimized
    :type variable: int
    :param value: the best value found so far for that variable
    :type value: int
    :param bound: the side of its domain the bound tightens
    :type bound: int

    :return: the events triggered by the tightening, EVENT_MASK_NONE when it changed nothing,
             -1 when the choice point wipes out
    :rtype: int
    """
    cell_idx = variable << 1
    # strictly better than the best value so far
    better = value + 1 if bound == DOMAIN_MIN else value - 1
    new_min, new_max = bounds_after_tightening(state[cell_idx], state[cell_idx | 1], bound, better)
    if new_min > new_max:
        return -1
    events, utrail_size = tighten_at(state, utrail_log, utrail_indices, mark, utrail_top[0], variable, new_min, new_max)
    utrail_top[0] = utrail_size
    return events


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    state: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
    entailed: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    priorities: NDArray,
    bb: NDArray,
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
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray
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
    :param bb: the branch-and-bound bound as a Numpy array of variable, bound and value,
               whose variable is -1 when not optimizing
    :type bb: NDArray

    :return: true iff it is possible to backtrack
    :rtype: bool
    """
    optimized_variable = bb[BB_VARIABLE]
    while choicepoint_top[0] > 0:
        choicepoint_top[0] -= 1
        choicepoint = choicepoint_top[0]
        statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
        mark = choicepoints[choicepoint, CHOICEPOINT_UTRAIL_MARK]
        utrail_undo(state, utrail_log, utrail_indices, utrail_top, mark)
        variable = choicepoints[choicepoint, CHOICEPOINT_VARIABLE]
        cell_idx = variable << 1
        new_min, new_max = bounds_after_tightening(
            state[cell_idx],
            state[cell_idx | 1],
            choicepoints[choicepoint, CHOICEPOINT_BOUND],
            choicepoints[choicepoint, CHOICEPOINT_VALUE],
        )
        events, utrail_size = tighten_at(
            state, utrail_log, utrail_indices, mark, utrail_top[0], variable, new_min, new_max
        )
        utrail_top[0] = utrail_size
        if events != EVENT_MASK_NONE:
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                priorities,
                variable,
                events,
                -1,
            )
        if optimized_variable < 0:
            return True
        events = tighten_objective(
            state,
            utrail_log,
            utrail_top,
            utrail_indices,
            mark,
            optimized_variable,
            bb[BB_VALUE],
            bb[BB_BOUND],
        )
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
                -1,
            )
        return True
    return False
