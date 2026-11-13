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
"""
The decisions of lazy clause generation as literals (design/proposals/lcg.md, 4.2).

Each decision is one literal on its own choice point. The domain heuristics stay as they are, and their results map
to literals (decision_literal):

- DECISION_LE at v gives [x <= v], and DECISION_GT at v gives [x >= v + 1];
- DECISION_EQ at the min or at the max of the domain gives one literal, [x <= v] or [x >= v];
- DECISION_EQ at an interior v gives two literals on two choice points: [x <= v] first, then [x >= v].

The second literal of an interior EQ waits in pending_lit while the first one propagates: decision_literal writes it
there, and take_pending_lit reads it. If the propagation makes it true (x is fixed to v) or false (the max of x is now
below v), no choice point opens for it.

The order of the two literals keeps the order of the values of BacktrackSolver: v first, then the values below v,
then the values above v. Each choice point parks the negation of its literal, so a backtrack to the second one
applies [x <= v - 1], and a backtrack to the first one applies [x >= v + 1].
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DECISION_EQ, DECISION_LE, DOMAIN_MAX, DOMAIN_MIN
from nucs.solvers.choicepoints import CHOICEPOINT_UTRAIL_MARK, bounds_after_tightening, park_alternative
from nucs.solvers.lcg.literals import lit_is_unassigned, lit_negate
from nucs.solvers.state import tighten_at

# The cells of pending_lit, the second literal of an interior EQ: its var_side, -1 when there is none, and its value.
PENDING_LIT_VAR_SIDE = 0
PENDING_LIT_VALUE = 1


@njit(cache=True, inline="always")
def decision_literal(
    domain_min: int, domain_max: int, variable: int, kind: int, value: int, pending_lit: NDArray
) -> tuple[int, int]:
    """
    Returns the literal of a decision of a domain heuristic, and writes the second literal of an interior EQ in
    pending_lit.

    An EQ value at or outside a bound of the domain gives one literal, at that bound, as branch does when it clamps the
    value: min_cost returns -1 when no value of the domain has a positive cost.

    :param domain_min: the min of the domain of the variable
    :type domain_min: int
    :param domain_max: the max of the domain of the variable
    :type domain_max: int
    :param variable: the variable of the decision
    :type variable: int
    :param kind: the kind of split, one of DECISION_LE, DECISION_GT and DECISION_EQ
    :type kind: int
    :param value: the value the domain is split at
    :type value: int
    :param pending_lit: the second literal of an interior EQ, see PENDING_LIT_*; it does not change for the other
                        decisions
    :type pending_lit: NDArray

    :return: the var_side and the value of the literal to apply now
    :rtype: Tuple[int, int]
    """
    if kind == DECISION_EQ:
        if value >= domain_max:
            return (variable << 1) | DOMAIN_MIN, domain_max
        if value <= domain_min:
            return (variable << 1) | DOMAIN_MAX, domain_min
        pending_lit[PENDING_LIT_VAR_SIDE] = (variable << 1) | DOMAIN_MIN
        pending_lit[PENDING_LIT_VALUE] = value
        return (variable << 1) | DOMAIN_MAX, value
    if kind == DECISION_LE:
        return (variable << 1) | DOMAIN_MAX, value
    return (variable << 1) | DOMAIN_MIN, value + 1


@njit(cache=True, inline="always")
def take_pending_lit(state: NDArray, pending_lit: NDArray) -> tuple[int, int]:
    """
    Returns the second literal of an interior EQ when it is still a decision to make, and clears it.

    :param state: all the backtrackable state, whose head holds the bounds of the domains
    :type state: NDArray
    :param pending_lit: the second literal of an interior EQ, see PENDING_LIT_*
    :type pending_lit: NDArray

    :return: the var_side and the value of the literal, or a var_side of -1 when there is none, or when the
             propagation of the first literal made it true or false
    :rtype: Tuple[int, int]
    """
    var_side = pending_lit[PENDING_LIT_VAR_SIDE]
    value = pending_lit[PENDING_LIT_VALUE]
    pending_lit[PENDING_LIT_VAR_SIDE] = -1
    if var_side < 0 or not lit_is_unassigned(state, var_side, value):
        return -1, 0
    return var_side, value


@njit(cache=True)
def decide(
    state: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
    var_side: int,
    value: int,
) -> int:
    """
    Applies a decision literal on a new choice point, and parks its negation for the search to resume.

    As in branch, the choice point records the undo trail position and the alternative, and the choice point above it
    gets the same mark, because it is where the search is about to work.

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
    :param var_side: the bound of the decision literal, (variable << 1) | side
    :type var_side: int
    :param value: the value of the decision literal
    :type value: int

    :return: the events that the decision triggers
    :rtype: int
    """
    choicepoint = choicepoint_top[0]
    mark = utrail_top[0]
    variable = var_side >> 1
    negation_var_side, negation_value = lit_negate(var_side, value)
    park_alternative(choicepoints, choicepoint, mark, variable, negation_var_side & 1, negation_value)
    choicepoints[choicepoint + 1, CHOICEPOINT_UTRAIL_MARK] = mark
    choicepoint_top[0] = choicepoint + 1
    cell_idx = variable << 1
    new_min, new_max = bounds_after_tightening(state[cell_idx], state[cell_idx | 1], var_side & 1, value)
    # the undo trail size is the mark: the choice point has just been pushed at it
    events, utrail_sz = tighten_at(state, utrail_log, utrail_indices, mark, mark, variable, new_min, new_max)
    utrail_top[0] = utrail_sz
    return events
