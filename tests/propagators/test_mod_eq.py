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
from typing import List, Optional, Tuple, Union

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.mod_eq_propagator import compute_domains_mod_eq
from tests.propagators.propagator_test import PropagatorTest


class TestModEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # both operands fixed: the truncated remainder takes the sign of the dividend x
            ([(7, 7), (3, 3), (0, 10)], [], PROP_ENTAILMENT, [[7, 7], [3, 3], [1, 1]]),
            ([(-7, -7), (3, 3), (-10, 10)], [], PROP_ENTAILMENT, [[-7, -7], [3, 3], [-1, -1]]),
            ([(7, 7), (-3, -3), (-10, 10)], [], PROP_ENTAILMENT, [[7, 7], [-3, -3], [1, 1]]),
            ([(-7, -7), (-3, -3), (-10, 10)], [], PROP_ENTAILMENT, [[-7, -7], [-3, -3], [-1, -1]]),
            ([(2, 2), (3, 3), (2, 2)], [], PROP_ENTAILMENT, [[2, 2], [3, 3], [2, 2]]),
            # fixed operands but the remainder is outside z
            ([(7, 7), (3, 3), (2, 2)], [], PROP_INCONSISTENCY, None),
            # z is bounded by |y| - 1 and by the sign of x
            ([(0, 9), (3, 3), (-10, 10)], [], PROP_CONSISTENCY, [[0, 9], [3, 3], [0, 2]]),
            ([(0, 100), (1, 1), (0, 0)], [], PROP_CONSISTENCY, [[0, 100], [1, 1], [0, 0]]),
            # x <= 0 forces z <= 0, so z collapses to 0 here
            ([(-5, -1), (2, 4), (0, 10)], [], PROP_CONSISTENCY, [[-5, -1], [2, 4], [0, 0]]),
            # |y| > |z|: a non-zero z lower-bounds the magnitude of a sign-definite y
            ([(-100, 100), (2, 100), (5, 5)], [], PROP_CONSISTENCY, [[-100, 100], [6, 100], [5, 5]]),
            # y cannot be zero
            ([(5, 5), (0, 0), (0, 0)], [], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_mod_eq, domains, parameters, consistency_result, expected_domains)
