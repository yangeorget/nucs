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
import itertools

import numpy as np
import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.solvers.lcg.literals import lit_is_false, lit_is_true, lit_negate


class TestLiterals:
    @pytest.mark.parametrize(
        "variable,side,value,negation",
        [
            (0, DOMAIN_MIN, 5, (1, 4)),  # [x0 >= 5] -> [x0 <= 4]
            (0, DOMAIN_MAX, 5, (0, 6)),  # [x0 <= 5] -> [x0 >= 6]
            (3, DOMAIN_MIN, -2, (7, -3)),  # [x3 >= -2] -> [x3 <= -3]
            (3, DOMAIN_MAX, -2, (6, -1)),  # [x3 <= -2] -> [x3 >= -1]
            (2, DOMAIN_MIN, 1, (5, 0)),  # a 0/1 variable: [b >= 1] (true) -> [b <= 0] (false)
        ],
    )
    def test_lit_negate(self, variable: int, side: int, value: int, negation: tuple[int, int]) -> None:
        var_side = (variable << 1) | side
        assert lit_negate(var_side, value) == negation
        assert lit_negate(*negation) == (var_side, value)

    @pytest.mark.parametrize(
        "side,value,is_true,is_false",
        [
            (DOMAIN_MIN, 2, True, False),  # [x >= 2], the lower bound itself
            (DOMAIN_MIN, 1, True, False),  # [x >= 1], weaker than the lower bound
            (DOMAIN_MAX, 1, False, True),  # [x <= 1], below the domain
            (DOMAIN_MIN, 5, False, False),  # [x >= 5], inside the domain: unassigned
            (DOMAIN_MAX, 7, True, False),  # [x <= 7], the upper bound itself
            (DOMAIN_MIN, 8, False, True),  # [x >= 8], above the domain
        ],
    )
    def test_lit_is_true_is_false(self, side: int, value: int, is_true: bool, is_false: bool) -> None:
        # the example of lcg.md 1.2: x (variable 1) in [2, 7], after a variable 0 in [0, 9]
        state = np.array([0, 9, 2, 7], dtype=np.int32)
        var_side = (1 << 1) | side
        assert lit_is_true(state, var_side, value) == is_true
        assert lit_is_false(state, var_side, value) == is_false

    def test_soundness_against_brute_force(self) -> None:
        # every box [lo, hi] in [-3, 3] and every literal with a value in [-5, 5]: true must mean that all the values
        # of the box satisfy the literal, false that none does; the values outside the box test the root bounds
        state = np.array([0, 9, 0, 0], dtype=np.int32)
        for lo, hi in itertools.combinations_with_replacement(range(-3, 4), 2):
            state[2], state[3] = lo, hi
            for side, value in itertools.product((DOMAIN_MIN, DOMAIN_MAX), range(-5, 6)):
                var_side = (1 << 1) | side
                sat = [x >= value if side == DOMAIN_MIN else x <= value for x in range(lo, hi + 1)]
                is_true = lit_is_true(state, var_side, value)
                is_false = lit_is_false(state, var_side, value)
                literal = f"side={side} value={value} box=[{lo}, {hi}]"
                assert is_true == all(sat), literal
                assert is_false == (not any(sat)), literal
                # a literal on a fixed variable is never unassigned
                assert lo < hi or is_true != is_false, literal
                # the negation is false exactly when the literal is true
                assert lit_is_false(state, *lit_negate(var_side, value)) == is_true, literal
