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
import numpy as np
import pytest
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_GROUND,
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    EVENT_MASK_MIN_MAX,
    EVENT_MASK_NONE,
)
from nucs.solvers.lcg.implication_trail import (
    LIT_LEVEL,
    LIT_PREV,
    LIT_REASON_DATA,
    LIT_REASON_KIND,
    LIT_REASON_POS,
    LIT_TRAIL_VALID,
    LIT_VALUE,
    LIT_VAR_SIDE,
    LIT_WIDTH,
    REASON_DECISION,
    REASON_PROPAGATOR,
    REASON_ROOT,
    allocate_implication_trail,
    lit_pop,
    lit_push,
    lit_push_events,
    lit_trail_check,
    lit_trail_root,
)

INITIAL_DOMAINS = np.array([[0, 9], [0, 9]], dtype=np.int32)


class TestImplicationTrail:
    def test_lit_push_and_lit_pop_keep_the_chains(self) -> None:
        lit_trail, lit_top, lit_last, _ = allocate_implication_trail(2, 4)
        # two changes of the min of x0, with a change of the max of x1 between them
        lit_size = lit_push(lit_trail, lit_last, 0, 0, 2, 0, REASON_PROPAGATOR, 7, 0)
        lit_size = lit_push(lit_trail, lit_last, lit_size, 3, 5, 1, REASON_DECISION, -1, -1)
        lit_size = lit_push(lit_trail, lit_last, lit_size, 0, 4, 1, REASON_PROPAGATOR, 8, 2)
        lit_top[0] = lit_size
        assert lit_trail[:3].tolist() == [
            [0, 2, -1, 0, REASON_PROPAGATOR, 7, 0],
            [3, 5, -1, 1, REASON_DECISION, -1, -1],
            [0, 4, 0, 1, REASON_PROPAGATOR, 8, 2],  # PREV is the first change of the min of x0
        ]
        assert lit_last.tolist() == [2, -1, -1, 1]
        lit_pop(lit_trail, lit_top, lit_last, 1)
        # the min of x0 is back at its first change, the max of x1 keeps its row
        assert lit_top.tolist() == [1]
        assert lit_last.tolist() == [0, -1, -1, -1]
        lit_pop(lit_trail, lit_top, lit_last, 0)
        assert lit_top.tolist() == [0]
        assert lit_last.tolist() == [-1, -1, -1, -1]

    @pytest.mark.parametrize(
        "events,expected_rows",
        [
            (EVENT_MASK_NONE, []),
            (EVENT_MASK_MIN, [[2, 3]]),
            (EVENT_MASK_MAX, [[3, 6]]),
            (EVENT_MASK_MIN_MAX, [[2, 3], [3, 6]]),
            # the ground event adds no row: it is not a bound
            (EVENT_MASK_MIN_MAX | EVENT_MASK_GROUND, [[2, 3], [3, 6]]),
        ],
    )
    def test_lit_push_events_adds_a_row_for_each_moved_bound(self, events: int, expected_rows: list[list[int]]) -> None:
        lit_trail, _, lit_last, _ = allocate_implication_trail(2, 4)
        state = np.array([0, 9, 3, 6, 2], dtype=np.int32)  # x1 is now [3, 6]
        lit_size = lit_push_events(lit_trail, lit_last, 0, state, 1, events, 2, REASON_PROPAGATOR, 5, 0)
        assert lit_size == len(expected_rows)
        assert lit_trail[:lit_size, [LIT_VAR_SIDE, LIT_VALUE]].tolist() == expected_rows
        assert lit_trail[:lit_size, [LIT_LEVEL, LIT_REASON_KIND, LIT_REASON_DATA, LIT_REASON_POS]].tolist() == [
            [2, REASON_PROPAGATOR, 5, 0]
        ] * len(expected_rows)

    def test_lit_trail_root_records_the_bounds_not_at_their_initial_value(self) -> None:
        lit_trail, lit_top, lit_last, level_start = allocate_implication_trail(2, 4)
        # what a previous search left, and the bound of the objective x1 at the root
        lit_top[0] = 5
        lit_last[:] = 4
        level_start[0] = 3
        state = np.array([0, 9, 0, 6, 2], dtype=np.int32)
        lit_trail_root(state, INITIAL_DOMAINS, lit_trail, lit_top, lit_last, level_start)
        assert lit_top.tolist() == [1]
        assert lit_trail[0].tolist() == [3, 6, -1, 0, REASON_ROOT, -1, -1]
        assert lit_last.tolist() == [-1, -1, -1, 0]
        assert level_start[0] == 0
        assert lit_trail_check(state, INITIAL_DOMAINS, lit_trail, lit_top, lit_last, level_start, 0) == LIT_TRAIL_VALID

    def test_lit_trail_check_accepts_a_valid_trail(self) -> None:
        state, lit_trail, lit_top, lit_last, level_start = build_valid_trail()
        assert lit_trail_check(state, INITIAL_DOMAINS, lit_trail, lit_top, lit_last, level_start, 1) == LIT_TRAIL_VALID

    @pytest.mark.parametrize(
        "row,column,value,expected",
        [
            (2, LIT_VALUE, 2, 2),  # a row that does not tighten its bound
            (2, LIT_PREV, -1, 2),  # a broken chain
            (1, LIT_LEVEL, 0, 1),  # a row of level 0 after the start of level 1
            (2, LIT_LEVEL, 2, 2),  # a level above the current one
            (2, LIT_REASON_POS, 0, 2),  # a call that starts in an earlier level
            (2, LIT_REASON_POS, 3, 2),  # a call that starts after its own row
        ],
    )
    def test_lit_trail_check_finds_the_first_bad_row(self, row: int, column: int, value: int, expected: int) -> None:
        state, lit_trail, lit_top, lit_last, level_start = build_valid_trail()
        lit_trail[row, column] = value
        assert lit_trail_check(state, INITIAL_DOMAINS, lit_trail, lit_top, lit_last, level_start, 1) == expected

    @pytest.mark.parametrize(
        "array,index,value,expected",
        [
            ("state", 3, 6, 3),  # the replay does not give the state: the number of rows
            ("lit_last", 1, 0, 3),  # a chain that does not end at the row of lit_last
            ("level_start", 0, 1, 0),  # level 0 does not start at row 0
        ],
    )
    def test_lit_trail_check_compares_with_the_state(self, array: str, index: int, value: int, expected: int) -> None:
        state, lit_trail, lit_top, lit_last, level_start = build_valid_trail()
        {"state": state, "lit_last": lit_last, "level_start": level_start}[array][index] = value
        assert lit_trail_check(state, INITIAL_DOMAINS, lit_trail, lit_top, lit_last, level_start, 1) == expected

    @pytest.mark.parametrize(
        "domain_nb,level_nb,row_nb",
        [
            (2, 8, 1 << 16),  # a small problem gets the floor of the undo trail
            (3000, 12000, 32 * 3000),  # a wide problem gets 16 changes of each bound
        ],
    )
    def test_allocate_implication_trail(self, domain_nb: int, level_nb: int, row_nb: int) -> None:
        lit_trail, lit_top, lit_last, level_start = allocate_implication_trail(domain_nb, level_nb)
        assert lit_trail.shape == (row_nb, LIT_WIDTH)
        assert lit_top.tolist() == [0]
        assert lit_last.tolist() == [-1] * (2 * domain_nb)
        assert level_start.shape == (level_nb,)


def build_valid_trail() -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Builds the implication trail of a path to level 1 on two variables in [0, 9]: at level 0, a propagator sets
    [x0 >= 2]; the decision [x1 <= 5] opens level 1; a propagator then sets [x0 >= 3].

    :return: the state, the implication trail, the number of rows in use, the last row of each bound and the first row
             of each level
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]
    """
    lit_trail, lit_top, lit_last, level_start = allocate_implication_trail(2, 4)
    lit_size = lit_push(lit_trail, lit_last, 0, 0 << 1 | DOMAIN_MIN, 2, 0, REASON_PROPAGATOR, 0, 0)
    level_start[1] = lit_size
    lit_size = lit_push(lit_trail, lit_last, lit_size, 1 << 1 | DOMAIN_MAX, 5, 1, REASON_DECISION, -1, -1)
    lit_size = lit_push(lit_trail, lit_last, lit_size, 0 << 1 | DOMAIN_MIN, 3, 1, REASON_PROPAGATOR, 1, 2)
    lit_top[0] = lit_size
    state = np.array([3, 9, 0, 5, 2], dtype=np.int32)
    return state, lit_trail, lit_top, lit_last, level_start
