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
The implication trail of lazy clause generation (design/proposals/lcg.md, 3.3 and 3.4).

The implication trail records each bound change on the current path, one row for each change, with its choice point
and its reason. It is not the undo trail: the undo trail skips a write to a cell that it already saved at the current
choice point, but the conflict analysis needs every change. The undo trail still restores state; the implication
trail only records.

A row means "the literal (VAR_SIDE, VALUE) became true here". The rows of one bound make a chain through PREV, and
itrail_last holds the last row of each chain, -1 when the bound has its root value.

The choice point of a row is choicepoint_top[0] when the row is written: in the terms of the conflict analysis, its
decision level. When choice point c branches, the solver writes the size of the implication trail in its column
CHOICEPOINT_ITRAIL_MARK, as branch writes the size of the undo trail in CHOICEPOINT_UTRAIL_MARK. Thus the rows of
choice point c start at the mark of choice point c - 1 (itrail_start), and a backtrack to choice point c pops the rows
down to the mark of c, as it undoes the undo trail down to the other mark of c.
"""

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MAX, EVENT_MASK_MIN
from nucs.solvers.choicepoints import CHOICEPOINT_ITRAIL_MARK

# The columns of a row of itrail.
ITRAIL_VAR_SIDE = 0  # the bound that changed, (variable << 1) | side, which is also its index in state
ITRAIL_VALUE = 1  # the new value of the bound
ITRAIL_PREV = 2  # the row of the previous change of the same bound, -1 when there is none
ITRAIL_CHOICEPOINT = 3  # the choice point of the change, its decision level
ITRAIL_REASON_KIND = 4  # why the bound changed, one of REASON_*
ITRAIL_REASON_DATA = 5  # the propagator for REASON_PROPAGATOR, the clause for REASON_CLAUSE, -1 otherwise
ITRAIL_REASON_POS = 6  # for REASON_PROPAGATOR, the first row of the propagator call, -1 otherwise
ITRAIL_WIDTH = 7

# The kinds of reasons.
REASON_DECISION = 0  # a decision of the search, or a parked alternative that a backtrack applies
REASON_PROPAGATOR = 1  # a propagator made the change
REASON_CLAUSE = 2  # a learned clause made the change (from step 9)
REASON_ROOT = 3  # a fact that holds for the rest of the search, such as the bound of the best objective

# The rows that a step of the search can write outside the propagation: a decision moves at most two bounds, and a
# backtrack moves one bound for the parked alternative and one bound for the objective.
ITRAIL_STEP_ROW_NB = 2

# What the debug replay check returns when the implication trail agrees with state.
ITRAIL_VALID = -1


@njit(cache=True, inline="always")
def itrail_push(
    itrail: NDArray,
    itrail_last: NDArray,
    itrail_size: int,
    var_side: int,
    value: int,
    choicepoint: int,
    reason_kind: int,
    reason_data: int,
    reason_pos: int,
) -> int:
    """
    Adds a row to the implication trail and makes it the last row of the chain of its bound.

    The caller makes sure that the row fits: nothing here tests the size of itrail.

    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param itrail_size: the number of rows in use
    :type itrail_size: int
    :param var_side: the bound that changed
    :type var_side: int
    :param value: the new value of the bound
    :type value: int
    :param choicepoint: the choice point of the change, choicepoint_top[0]
    :type choicepoint: int
    :param reason_kind: why the bound changed, one of REASON_*
    :type reason_kind: int
    :param reason_data: the propagator or the clause of the reason, -1 when there is none
    :type reason_data: int
    :param reason_pos: the first row of the propagator call, -1 when the reason is not a propagator
    :type reason_pos: int

    :return: the new number of rows in use
    :rtype: int
    """
    itrail[itrail_size, ITRAIL_VAR_SIDE] = var_side
    itrail[itrail_size, ITRAIL_VALUE] = value
    itrail[itrail_size, ITRAIL_PREV] = itrail_last[var_side]
    itrail[itrail_size, ITRAIL_CHOICEPOINT] = choicepoint
    itrail[itrail_size, ITRAIL_REASON_KIND] = reason_kind
    itrail[itrail_size, ITRAIL_REASON_DATA] = reason_data
    itrail[itrail_size, ITRAIL_REASON_POS] = reason_pos
    itrail_last[var_side] = itrail_size
    return itrail_size + 1


@njit(cache=True, inline="always")
def itrail_push_events(
    itrail: NDArray,
    itrail_last: NDArray,
    itrail_size: int,
    state: NDArray,
    variable: int,
    events: int,
    choicepoint: int,
    reason_kind: int,
    reason_data: int,
    reason_pos: int,
) -> int:
    """
    Adds a row for each bound that a call of tighten_at moved, with the new value that the call wrote into state.

    The events tell which bounds moved: tighten_at sets EVENT_MASK_MIN only when the min changed, and EVENT_MASK_MAX
    only when the max changed.

    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param itrail_size: the number of rows in use
    :type itrail_size: int
    :param state: all the backtrackable state, which holds the new bounds
    :type state: NDArray
    :param variable: the variable that tighten_at wrote
    :type variable: int
    :param events: the events that tighten_at returned
    :type events: int
    :param choicepoint: the choice point of the change, choicepoint_top[0]
    :type choicepoint: int
    :param reason_kind: why the bounds changed, one of REASON_*
    :type reason_kind: int
    :param reason_data: the propagator or the clause of the reason, -1 when there is none
    :type reason_data: int
    :param reason_pos: the first row of the propagator call, -1 when the reason is not a propagator
    :type reason_pos: int

    :return: the new number of rows in use
    :rtype: int
    """
    cell_idx = variable << 1
    if events & EVENT_MASK_MIN:
        itrail_size = itrail_push(
            itrail,
            itrail_last,
            itrail_size,
            cell_idx,
            state[cell_idx],
            choicepoint,
            reason_kind,
            reason_data,
            reason_pos,
        )
    if events & EVENT_MASK_MAX:
        itrail_size = itrail_push(
            itrail,
            itrail_last,
            itrail_size,
            cell_idx | 1,
            state[cell_idx | 1],
            choicepoint,
            reason_kind,
            reason_data,
            reason_pos,
        )
    return itrail_size


@njit(cache=True, inline="always")
def itrail_pop(itrail: NDArray, itrail_top: NDArray, itrail_last: NDArray, itrail_mark: int) -> None:
    """
    Removes the rows from the top of the implication trail down to a mark, and sets itrail_last back along the chains.

    The cost is one step for each removed row, as for the undo trail.

    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param itrail_mark: the number of rows to keep
    :type itrail_mark: int
    """
    row = itrail_top[0]
    while row > itrail_mark:
        row -= 1
        itrail_last[itrail[row, ITRAIL_VAR_SIDE]] = itrail[row, ITRAIL_PREV]
    itrail_top[0] = itrail_mark


@njit(cache=True, inline="always")
def itrail_start(choicepoint_stk: NDArray, choicepoint: int) -> int:
    """
    Returns the first row of the implication trail that a choice point wrote.

    The rows of choice point c start when choice point c - 1 branches, and the rows of choice point 0 start at row 0.

    :param choicepoint_stk: the per-choice-point metadata
    :type choicepoint_stk: NDArray
    :param choicepoint: the choice point, at most choicepoint_top[0]
    :type choicepoint: int

    :return: the first row of the implication trail that the choice point wrote
    :rtype: int
    """
    return 0 if choicepoint == 0 else choicepoint_stk[choicepoint - 1, CHOICEPOINT_ITRAIL_MARK]


@njit(cache=True)
def itrail_root(
    state: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
) -> None:
    """
    Builds the implication trail again for a root on which no propagation has run yet.

    The solver calls this after each return to the root, and after it applies the bound of the best objective there.
    The implication trail is then empty, except for one row with REASON_ROOT for each bound that is not at its
    initial value: at this point, that can only be the bound of the objective.

    :param state: all the backtrackable state
    :type state: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    """
    itrail_last.fill(-1)
    itrail_size = 0
    for variable in range(len(initial_domains)):
        for side in range(2):
            var_side = (variable << 1) | side
            if state[var_side] != initial_domains[variable, side]:
                itrail_size = itrail_push(
                    itrail, itrail_last, itrail_size, var_side, state[var_side], 0, REASON_ROOT, -1, -1
                )
    itrail_top[0] = itrail_size


@njit(cache=True)
def itrail_check(
    state: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    choicepoint_stk: NDArray,
    choicepoint: int,
) -> int:
    """
    Replays the implication trail from the root, and checks it against state: the debug check of step 5 of
    design/proposals/lcg.md 7.0.

    The checks are:

    - each row makes its bound strictly tighter than the replay so far, and its PREV is the last row of its bound;
    - the choice points of the rows do not decrease and are at most the current one, and each row is between the
      first row of its choice point (itrail_start) and the mark of its choice point, when that one has branched;
    - the REASON_POS of a propagator row is a row of the same choice point, and not after the row;
    - at the end, the replay gives the bounds of state, and the chains end at the rows of itrail_last.

    :param state: all the backtrackable state
    :type state: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param choicepoint_stk: the per-choice-point metadata, with the marks of the implication trail
    :type choicepoint_stk: NDArray
    :param choicepoint: the current choice point, choicepoint_top[0]
    :type choicepoint: int

    :return: ITRAIL_VALID when the implication trail agrees with state, else the first row that does not, or the
             number of rows in use when only the final comparison fails
    :rtype: int
    """
    itrail_size = itrail_top[0]
    replay = np.empty(2 * len(initial_domains), dtype=np.int32)
    last = np.full(2 * len(initial_domains), -1, dtype=np.int32)
    for variable in range(len(initial_domains)):
        replay[variable << 1] = initial_domains[variable, DOMAIN_MIN]
        replay[(variable << 1) | 1] = initial_domains[variable, DOMAIN_MAX]
    previous_choicepoint = 0
    for row in range(itrail_size):
        var_side = itrail[row, ITRAIL_VAR_SIDE]
        value = itrail[row, ITRAIL_VALUE]
        row_choicepoint = itrail[row, ITRAIL_CHOICEPOINT]
        if itrail[row, ITRAIL_PREV] != last[var_side]:
            return row
        if (var_side & 1) == DOMAIN_MIN:
            if value <= replay[var_side]:
                return row
        elif value >= replay[var_side]:
            return row
        start = itrail_start(choicepoint_stk, row_choicepoint)
        if row_choicepoint < previous_choicepoint or row_choicepoint > choicepoint or row < start:
            return row
        if row_choicepoint < choicepoint and row >= choicepoint_stk[row_choicepoint, CHOICEPOINT_ITRAIL_MARK]:
            return row
        if itrail[row, ITRAIL_REASON_KIND] == REASON_PROPAGATOR and not (
            start <= itrail[row, ITRAIL_REASON_POS] <= row
        ):
            return row
        replay[var_side] = value
        last[var_side] = row
        previous_choicepoint = row_choicepoint
    for var_side in range(len(replay)):
        if replay[var_side] != state[var_side] or last[var_side] != itrail_last[var_side]:
            return itrail_size
    return ITRAIL_VALID


def allocate_itrail(domain_nb: int) -> tuple[NDArray, NDArray, NDArray]:
    """
    Allocates the implication trail of a problem.

    :param domain_nb: the number of variables of the problem
    :type domain_nb: int

    :return: the implication trail; the number of rows in use, as a one-cell array; the last row of each bound, -1
             when the bound has its root value
    :rtype: Tuple[NDArray, NDArray, NDArray]
    """
    # The implication trail holds only the changes on the current path. The floor is that of the undo trail; the
    # second term gives a wide model 16 changes of each bound before the first growth.
    itrail = np.empty((max(1 << 16, 32 * domain_nb), ITRAIL_WIDTH), dtype=np.int32)
    itrail_top = np.zeros((1,), dtype=np.int32)
    itrail_last = np.full(2 * domain_nb, -1, dtype=np.int32)
    return itrail, itrail_top, itrail_last
