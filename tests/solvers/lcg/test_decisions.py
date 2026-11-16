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

from nucs.constants import (
    DECISION_EQ,
    DECISION_GT,
    DECISION_LE,
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_GROUND,
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
)
from nucs.solvers.choicepoints import (
    CHOICEPOINT_BOUND,
    CHOICEPOINT_UTRAIL_MARK,
    CHOICEPOINT_VALUE,
    CHOICEPOINT_VARIABLE,
    CHOICEPOINT_WIDTH,
)
from nucs.solvers.lcg.decisions import (
    PENDING_LIT_VALUE,
    PENDING_LIT_VAR_SIDE,
    decide,
    decision_literal,
    take_pending_lit,
)


class TestDecisions:
    @pytest.mark.parametrize(
        "kind,value,expected,expected_pending_lit",
        [
            (DECISION_LE, 3, (1 << 1 | DOMAIN_MAX, 3), [-1, -1]),  # [x1 <= 3]
            (DECISION_GT, 3, (1 << 1 | DOMAIN_MIN, 4), [-1, -1]),  # [x1 >= 4]
            (DECISION_EQ, 0, (1 << 1 | DOMAIN_MAX, 0), [-1, -1]),  # an EQ at the min is [x1 <= min]
            (DECISION_EQ, 9, (1 << 1 | DOMAIN_MIN, 9), [-1, -1]),  # an EQ at the max is [x1 >= max]
            (DECISION_EQ, 4, (1 << 1 | DOMAIN_MAX, 4), [1 << 1 | DOMAIN_MIN, 4]),  # an interior EQ: [x1 >= 4] waits
            (DECISION_EQ, -1, (1 << 1 | DOMAIN_MAX, 0), [-1, -1]),  # min_cost with no positive cost: clamped to the min
            (DECISION_EQ, 12, (1 << 1 | DOMAIN_MIN, 9), [-1, -1]),  # clamped to the max
        ],
    )
    def test_decision_literal(
        self, kind: int, value: int, expected: tuple[int, int], expected_pending_lit: list[int]
    ) -> None:
        pending_lit = np.full(2, -1, dtype=np.int32)
        assert decision_literal(0, 9, 1, kind, value, pending_lit) == expected
        assert pending_lit.tolist() == expected_pending_lit

    @pytest.mark.parametrize(
        "pending_var_side,x1_domain,expected",
        [
            (-1, (0, 4), (-1, 0)),  # no pending literal
            (1 << 1 | DOMAIN_MIN, (0, 4), (1 << 1 | DOMAIN_MIN, 4)),  # [x1 >= 4] is still open: a decision to make
            (1 << 1 | DOMAIN_MIN, (4, 4), (-1, 0)),  # the propagation of [x1 <= 4] fixed x1 to 4: [x1 >= 4] is true
            (1 << 1 | DOMAIN_MIN, (0, 3), (-1, 0)),  # the propagation lowered the max below 4: [x1 >= 4] is false
        ],
    )
    def test_take_pending_lit(
        self, pending_var_side: int, x1_domain: tuple[int, int], expected: tuple[int, int]
    ) -> None:
        state = np.array([0, 9, *x1_domain, 2], dtype=np.int32)
        pending_lit = np.array([pending_var_side, 4], dtype=np.int32)
        assert take_pending_lit(state, pending_lit) == expected
        assert pending_lit[PENDING_LIT_VAR_SIDE] == -1  # taken or dropped, it does not stay
        assert pending_lit[PENDING_LIT_VALUE] == 4

    @pytest.mark.parametrize(
        "side,value,expected_x1_domain,expected_unbound_nb,expected_events,expected_alternative",
        [
            (DOMAIN_MAX, 4, [0, 4], 2, EVENT_MASK_MAX, [DOMAIN_MIN, 5]),  # [x1 <= 4] parks [x1 >= 5]
            (DOMAIN_MIN, 4, [4, 9], 2, EVENT_MASK_MIN, [DOMAIN_MAX, 3]),  # [x1 >= 4] parks [x1 <= 3]
            (
                DOMAIN_MAX,
                0,
                [0, 0],
                1,
                EVENT_MASK_MAX | EVENT_MASK_GROUND,
                [DOMAIN_MIN, 1],
            ),  # a literal that grounds x1
        ],
    )
    def test_decide_applies_the_literal_and_parks_its_negation(
        self,
        side: int,
        value: int,
        expected_x1_domain: list[int],
        expected_unbound_nb: int,
        expected_events: int,
        expected_alternative: list[int],
    ) -> None:
        state = np.array([0, 9, 0, 9, 2], dtype=np.int32)  # x0 and x1 in [0, 9], then the count of unbound variables
        utrail_log = np.zeros((8, 2), dtype=np.int32)
        utrail_top = np.array([3], dtype=np.int32)  # the undo trail already holds 3 entries: the mark is 3
        utrail_indices = np.full(len(state), -1, dtype=np.int32)
        choicepoints = np.zeros((4, CHOICEPOINT_WIDTH), dtype=np.int32)
        choicepoint_top = np.array([1], dtype=np.int32)
        events = decide(
            state, utrail_log, utrail_top, utrail_indices, choicepoints, choicepoint_top, 1 << 1 | side, value
        )
        assert events == expected_events
        assert state.tolist() == [0, 9, *expected_x1_domain, expected_unbound_nb]
        assert choicepoint_top.tolist() == [2]
        parked = choicepoints[1, [CHOICEPOINT_UTRAIL_MARK, CHOICEPOINT_VARIABLE, CHOICEPOINT_BOUND, CHOICEPOINT_VALUE]]
        assert parked.tolist() == [3, 1, *expected_alternative]
        assert choicepoints[2, CHOICEPOINT_UTRAIL_MARK] == 3  # the choice point where the search works now
