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

The implication trail records each bound change on the current path, one row for each change, with its decision level
and its reason. It is not the undo trail: the undo trail skips a write to a cell that it already saved at the current
choice point, but the conflict analysis needs every change. The undo trail still restores state; the implication
trail only records.

A row means "the literal (VAR_SIDE, VALUE) became true here". The rows of one bound make a chain through PREV, and
lit_last holds the last row of each chain, -1 when the bound has its root value. A backtrack pops the rows above the
first row of the level that it returns to, and sets lit_last back along the chains.
"""

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MAX, EVENT_MASK_MIN

# The columns of a row of lit_trail.
LIT_VAR_SIDE = 0  # the bound that changed, (variable << 1) | side, which is also its index in state
LIT_VALUE = 1  # the new value of the bound
LIT_PREV = 2  # the row of the previous change of the same bound, -1 when there is none
LIT_LEVEL = 3  # the decision level of the change
LIT_REASON_KIND = 4  # why the bound changed, one of REASON_*
LIT_REASON_DATA = 5  # the propagator for REASON_PROPAGATOR, the clause for REASON_CLAUSE, -1 otherwise
LIT_REASON_POS = 6  # for REASON_PROPAGATOR, the first row of the propagator call, -1 otherwise
LIT_WIDTH = 7

# The kinds of reasons.
REASON_DECISION = 0  # a decision of the search, or a parked alternative that a backtrack applies
REASON_PROPAGATOR = 1  # a propagator made the change
REASON_CLAUSE = 2  # a learned clause made the change (from step 9)
REASON_ROOT = 3  # a fact that holds for the rest of the search, such as the bound of the best objective

# The rows that a step of the search can write outside the propagation: a decision moves at most two bounds, and a
# backtrack moves one bound for the parked alternative and one bound for the objective.
LIT_STEP_ROW_NB = 2

# What the debug replay check returns when the implication trail agrees with state.
LIT_TRAIL_VALID = -1


@njit(cache=True, inline="always")
def lit_push(
    lit_trail: NDArray,
    lit_last: NDArray,
    lit_size: int,
    var_side: int,
    value: int,
    level: int,
    reason_kind: int,
    reason_data: int,
    reason_pos: int,
) -> int:
    """
    Adds a row to the implication trail and makes it the last row of the chain of its bound.

    The caller makes sure that the row fits: nothing here tests the size of lit_trail.

    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param lit_size: the number of rows in use
    :type lit_size: int
    :param var_side: the bound that changed
    :type var_side: int
    :param value: the new value of the bound
    :type value: int
    :param level: the decision level of the change
    :type level: int
    :param reason_kind: why the bound changed, one of REASON_*
    :type reason_kind: int
    :param reason_data: the propagator or the clause of the reason, -1 when there is none
    :type reason_data: int
    :param reason_pos: the first row of the propagator call, -1 when the reason is not a propagator
    :type reason_pos: int

    :return: the new number of rows in use
    :rtype: int
    """
    lit_trail[lit_size, LIT_VAR_SIDE] = var_side
    lit_trail[lit_size, LIT_VALUE] = value
    lit_trail[lit_size, LIT_PREV] = lit_last[var_side]
    lit_trail[lit_size, LIT_LEVEL] = level
    lit_trail[lit_size, LIT_REASON_KIND] = reason_kind
    lit_trail[lit_size, LIT_REASON_DATA] = reason_data
    lit_trail[lit_size, LIT_REASON_POS] = reason_pos
    lit_last[var_side] = lit_size
    return lit_size + 1


@njit(cache=True, inline="always")
def lit_push_events(
    lit_trail: NDArray,
    lit_last: NDArray,
    lit_size: int,
    state: NDArray,
    variable: int,
    events: int,
    level: int,
    reason_kind: int,
    reason_data: int,
    reason_pos: int,
) -> int:
    """
    Adds a row for each bound that a call of tighten_at moved, with the new value that the call wrote into state.

    The events tell which bounds moved: tighten_at sets EVENT_MASK_MIN only when the min changed, and EVENT_MASK_MAX
    only when the max changed.

    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param lit_size: the number of rows in use
    :type lit_size: int
    :param state: all the backtrackable state, which holds the new bounds
    :type state: NDArray
    :param variable: the variable that tighten_at wrote
    :type variable: int
    :param events: the events that tighten_at returned
    :type events: int
    :param level: the decision level of the change
    :type level: int
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
        lit_size = lit_push(
            lit_trail, lit_last, lit_size, cell_idx, state[cell_idx], level, reason_kind, reason_data, reason_pos
        )
    if events & EVENT_MASK_MAX:
        lit_size = lit_push(
            lit_trail,
            lit_last,
            lit_size,
            cell_idx | 1,
            state[cell_idx | 1],
            level,
            reason_kind,
            reason_data,
            reason_pos,
        )
    return lit_size


@njit(cache=True, inline="always")
def lit_pop(lit_trail: NDArray, lit_top: NDArray, lit_last: NDArray, lit_mark: int) -> None:
    """
    Removes the rows from the top of the implication trail down to a mark, and sets lit_last back along the chains.

    The cost is one step for each removed row, as for the undo trail.

    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_top: the number of rows in use, as a one-cell array
    :type lit_top: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param lit_mark: the number of rows to keep
    :type lit_mark: int
    """
    row = lit_top[0]
    while row > lit_mark:
        row -= 1
        lit_last[lit_trail[row, LIT_VAR_SIDE]] = lit_trail[row, LIT_PREV]
    lit_top[0] = lit_mark


@njit(cache=True)
def lit_trail_root(
    state: NDArray,
    initial_domains: NDArray,
    lit_trail: NDArray,
    lit_top: NDArray,
    lit_last: NDArray,
    level_start: NDArray,
) -> None:
    """
    Builds the implication trail again for a root on which no propagation has run yet.

    The solver calls this after each return to the root, and after it applies the bound of the best objective there.
    The trail is then empty, except for one row with REASON_ROOT for each bound that is not at its initial value:
    at this point, that can only be the bound of the objective.

    :param state: all the backtrackable state
    :type state: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_top: the number of rows in use, as a one-cell array
    :type lit_top: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param level_start: the first row of each decision level
    :type level_start: NDArray
    """
    lit_last.fill(-1)
    level_start[0] = 0
    lit_size = 0
    for variable in range(len(initial_domains)):
        for side in range(2):
            var_side = (variable << 1) | side
            if state[var_side] != initial_domains[variable, side]:
                lit_size = lit_push(lit_trail, lit_last, lit_size, var_side, state[var_side], 0, REASON_ROOT, -1, -1)
    lit_top[0] = lit_size


@njit(cache=True)
def lit_trail_check(
    state: NDArray,
    initial_domains: NDArray,
    lit_trail: NDArray,
    lit_top: NDArray,
    lit_last: NDArray,
    level_start: NDArray,
    level: int,
) -> int:
    """
    Replays the implication trail from the root, and checks it against state: the debug check of step 5 of
    design/proposals/lcg.md 7.0.

    The checks are:

    - each row makes its bound strictly tighter than the replay so far, and its PREV is the last row of its bound;
    - the levels of the rows do not decrease, are at most the current level, and each row is inside its level as
      level_start gives it;
    - the REASON_POS of a propagator row is in the same level, and not after the row;
    - at the end, the replay gives the bounds of state, and the chains end at the rows of lit_last.

    :param state: all the backtrackable state
    :type state: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_top: the number of rows in use, as a one-cell array
    :type lit_top: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param level_start: the first row of each decision level
    :type level_start: NDArray
    :param level: the current decision level, the height of the stack of choice points
    :type level: int

    :return: LIT_TRAIL_VALID when the trail agrees with state, else the first row that does not, or the number of
             rows in use when only the final comparison fails
    :rtype: int
    """
    lit_size = lit_top[0]
    replay = np.empty(2 * len(initial_domains), dtype=np.int32)
    last = np.full(2 * len(initial_domains), -1, dtype=np.int32)
    for variable in range(len(initial_domains)):
        replay[variable << 1] = initial_domains[variable, DOMAIN_MIN]
        replay[(variable << 1) | 1] = initial_domains[variable, DOMAIN_MAX]
    if level_start[0] != 0:
        return 0
    previous_level = 0
    for row in range(lit_size):
        var_side = lit_trail[row, LIT_VAR_SIDE]
        value = lit_trail[row, LIT_VALUE]
        row_level = lit_trail[row, LIT_LEVEL]
        if lit_trail[row, LIT_PREV] != last[var_side]:
            return row
        if (var_side & 1) == DOMAIN_MIN:
            if value <= replay[var_side]:
                return row
        elif value >= replay[var_side]:
            return row
        if row_level < previous_level or row_level > level or row < level_start[row_level]:
            return row
        if row_level < level and row >= level_start[row_level + 1]:
            return row
        if lit_trail[row, LIT_REASON_KIND] == REASON_PROPAGATOR and not (
            level_start[row_level] <= lit_trail[row, LIT_REASON_POS] <= row
        ):
            return row
        replay[var_side] = value
        last[var_side] = row
        previous_level = row_level
    for var_side in range(len(replay)):
        if replay[var_side] != state[var_side] or last[var_side] != lit_last[var_side]:
            return lit_size
    return LIT_TRAIL_VALID


def allocate_implication_trail(domain_nb: int, level_nb: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Allocates the implication trail of a problem.

    :param domain_nb: the number of variables of the problem
    :type domain_nb: int
    :param level_nb: the number of rows of the stack of choice points, which is also the number of decision levels
    :type level_nb: int

    :return: the implication trail; the number of rows in use, as a one-cell array; the last row of each bound, -1
             when the bound has its root value; the first row of each decision level
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray]
    """
    # The trail holds only the changes on the current path. The floor is that of the undo trail; the second term
    # gives a wide model 16 changes of each bound before the first growth.
    lit_trail = np.empty((max(1 << 16, 32 * domain_nb), LIT_WIDTH), dtype=np.int32)
    lit_top = np.zeros((1,), dtype=np.int32)
    lit_last = np.full(2 * domain_nb, -1, dtype=np.int32)
    level_start = np.zeros(level_nb, dtype=np.int32)
    return lit_trail, lit_top, lit_last, level_start
