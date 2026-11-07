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
from nucs.solvers.choice_points import CHOICE_POINT_ITRAIL_MARK, CHOICE_POINT_WIDTH
from nucs.solvers.lcg.implication_trail import (
    ITRAIL_CHOICE_POINT,
    ITRAIL_PREV,
    ITRAIL_REASON_DATA,
    ITRAIL_REASON_KIND,
    ITRAIL_REASON_POS,
    ITRAIL_VALID,
    ITRAIL_VALUE,
    ITRAIL_VAR_SIDE,
    ITRAIL_WIDTH,
    REASON_DECISION,
    REASON_PROPAGATOR,
    REASON_ROOT,
    allocate_itrail,
    itrail_check,
    itrail_pop,
    itrail_push,
    itrail_push_events,
    itrail_root,
    itrail_start,
)

INITIAL_DOMAINS = np.array([[0, 9], [0, 9]], dtype=np.int32)


class TestImplicationTrail:
    def test_itrail_push_and_itrail_pop_keep_the_chains(self) -> None:
        itrail, itrail_top, itrail_last = allocate_itrail(2)
        # two changes of the min of x0, with a change of the max of x1 between them
        itrail_size = itrail_push(itrail, itrail_last, 0, 0, 2, 0, REASON_PROPAGATOR, 7, 0)
        itrail_size = itrail_push(itrail, itrail_last, itrail_size, 3, 5, 1, REASON_DECISION, -1, -1)
        itrail_size = itrail_push(itrail, itrail_last, itrail_size, 0, 4, 1, REASON_PROPAGATOR, 8, 2)
        itrail_top[0] = itrail_size
        assert itrail[:3].tolist() == [
            [0, 2, -1, 0, REASON_PROPAGATOR, 7, 0],
            [3, 5, -1, 1, REASON_DECISION, -1, -1],
            [0, 4, 0, 1, REASON_PROPAGATOR, 8, 2],  # PREV is the first change of the min of x0
        ]
        assert itrail_last.tolist() == [2, -1, -1, 1]
        itrail_pop(itrail, itrail_top, itrail_last, 1)
        # the min of x0 is back at its first change, the max of x1 keeps its row
        assert itrail_top.tolist() == [1]
        assert itrail_last.tolist() == [0, -1, -1, -1]
        itrail_pop(itrail, itrail_top, itrail_last, 0)
        assert itrail_top.tolist() == [0]
        assert itrail_last.tolist() == [-1, -1, -1, -1]

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
    def test_itrail_push_events_adds_a_row_for_each_moved_bound(
        self, events: int, expected_rows: list[list[int]]
    ) -> None:
        itrail, _, itrail_last = allocate_itrail(2)
        state = np.array([0, 9, 3, 6, 2], dtype=np.int32)  # x1 is now [3, 6]
        itrail_size = itrail_push_events(itrail, itrail_last, 0, state, 1, events, 2, REASON_PROPAGATOR, 5, 0)
        assert itrail_size == len(expected_rows)
        assert itrail[:itrail_size, [ITRAIL_VAR_SIDE, ITRAIL_VALUE]].tolist() == expected_rows
        assert itrail[
            :itrail_size, [ITRAIL_CHOICE_POINT, ITRAIL_REASON_KIND, ITRAIL_REASON_DATA, ITRAIL_REASON_POS]
        ].tolist() == [[2, REASON_PROPAGATOR, 5, 0]] * len(expected_rows)

    def test_itrail_root_records_the_bounds_not_at_their_initial_value(self) -> None:
        itrail, itrail_top, itrail_last = allocate_itrail(2)
        # what a previous search left, and the bound of the objective x1 at the root
        itrail_top[0] = 5
        itrail_last[:] = 4
        state = np.array([0, 9, 0, 6, 2], dtype=np.int32)
        itrail_root(state, INITIAL_DOMAINS, itrail, itrail_top, itrail_last)
        assert itrail_top.tolist() == [1]
        assert itrail[0].tolist() == [3, 6, -1, 0, REASON_ROOT, -1, -1]
        assert itrail_last.tolist() == [-1, -1, -1, 0]
        choice_point_stk = np.zeros((4, CHOICE_POINT_WIDTH), dtype=np.int32)
        assert (
            itrail_check(state, INITIAL_DOMAINS, itrail, itrail_top, itrail_last, choice_point_stk, 0) == ITRAIL_VALID
        )

    def test_itrail_start_is_the_mark_of_the_previous_choice_point(self) -> None:
        choice_point_stk = np.zeros((4, CHOICE_POINT_WIDTH), dtype=np.int32)
        choice_point_stk[:, CHOICE_POINT_ITRAIL_MARK] = [3, 5, 5, 8]  # a ternary split parked 1 and 2 at row 5
        assert [itrail_start(choice_point_stk, choice_point) for choice_point in range(4)] == [0, 3, 5, 5]

    def test_itrail_check_accepts_a_valid_trail(self) -> None:
        state, itrail, itrail_top, itrail_last, choice_point_stk = build_valid_trail()
        assert (
            itrail_check(state, INITIAL_DOMAINS, itrail, itrail_top, itrail_last, choice_point_stk, 1) == ITRAIL_VALID
        )

    @pytest.mark.parametrize(
        "row,column,value,expected",
        [
            (2, ITRAIL_VALUE, 2, 2),  # a row that does not tighten its bound
            (2, ITRAIL_PREV, -1, 2),  # a broken chain
            (1, ITRAIL_CHOICE_POINT, 0, 1),  # a row of choice point 0 after choice point 0 branched
            (2, ITRAIL_CHOICE_POINT, 2, 2),  # a choice point above the current one
            (2, ITRAIL_REASON_POS, 0, 2),  # a call that starts at an earlier choice point
            (2, ITRAIL_REASON_POS, 3, 2),  # a call that starts after its own row
        ],
    )
    def test_itrail_check_finds_the_first_bad_row(self, row: int, column: int, value: int, expected: int) -> None:
        state, itrail, itrail_top, itrail_last, choice_point_stk = build_valid_trail()
        itrail[row, column] = value
        assert itrail_check(state, INITIAL_DOMAINS, itrail, itrail_top, itrail_last, choice_point_stk, 1) == expected

    @pytest.mark.parametrize(
        "array,index,value,expected",
        [
            ("state", 3, 6, 3),  # the replay does not give the state: the number of rows
            ("itrail_last", 1, 0, 3),  # a chain that does not end at the row of itrail_last
            ("choice_point_stk", (0, CHOICE_POINT_ITRAIL_MARK), 0, 0),  # choice point 0 branched before its own row
            ("choice_point_stk", (0, CHOICE_POINT_ITRAIL_MARK), 2, 1),  # choice point 0 branched after a row of 1
        ],
    )
    def test_itrail_check_compares_with_the_state_and_the_marks(
        self, array: str, index: int | tuple[int, int], value: int, expected: int
    ) -> None:
        state, itrail, itrail_top, itrail_last, choice_point_stk = build_valid_trail()
        {"state": state, "itrail_last": itrail_last, "choice_point_stk": choice_point_stk}[array][index] = value
        assert itrail_check(state, INITIAL_DOMAINS, itrail, itrail_top, itrail_last, choice_point_stk, 1) == expected

    @pytest.mark.parametrize(
        "domain_nb,row_nb",
        [
            (2, 1 << 16),  # a small problem gets the floor of the undo trail
            (3000, 32 * 3000),  # a wide problem gets 16 changes of each bound
        ],
    )
    def test_allocate_itrail(self, domain_nb: int, row_nb: int) -> None:
        itrail, itrail_top, itrail_last = allocate_itrail(domain_nb)
        assert itrail.shape == (row_nb, ITRAIL_WIDTH)
        assert itrail_top.tolist() == [0]
        assert itrail_last.tolist() == [-1] * (2 * domain_nb)


def build_valid_trail() -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Builds the implication trail of a path to choice point 1 on two variables in [0, 9]: at choice point 0, a
    propagator sets [x0 >= 2]; choice point 0 then branches on the decision [x1 <= 5]; at choice point 1, a propagator
    then sets [x0 >= 3].

    :return: the state, the implication trail, the number of rows in use, the last row of each bound and the stack of
             choice points
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]
    """
    itrail, itrail_top, itrail_last = allocate_itrail(2)
    choice_point_stk = np.zeros((4, CHOICE_POINT_WIDTH), dtype=np.int32)
    itrail_size = itrail_push(itrail, itrail_last, 0, 0 << 1 | DOMAIN_MIN, 2, 0, REASON_PROPAGATOR, 0, 0)
    choice_point_stk[0, CHOICE_POINT_ITRAIL_MARK] = itrail_size
    itrail_size = itrail_push(itrail, itrail_last, itrail_size, 1 << 1 | DOMAIN_MAX, 5, 1, REASON_DECISION, -1, -1)
    itrail_size = itrail_push(itrail, itrail_last, itrail_size, 0 << 1 | DOMAIN_MIN, 3, 1, REASON_PROPAGATOR, 1, 2)
    itrail_top[0] = itrail_size
    state = np.array([3, 9, 0, 5, 2], dtype=np.int32)
    return state, itrail, itrail_top, itrail_last, choice_point_stk
