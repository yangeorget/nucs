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

from nucs.constants import PROP_CONSISTENCY, PROP_INCONSISTENCY, PROP_ENTAILMENT
from nucs.propagators.mul_c_eq_propagator import compute_domains_mul_c_eq
from tests.propagators.propagator_test import PropagatorTest


class TestMulCEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # c > 0: y tightens from x
            ([(1, 5), (-100, 100)], [3], PROP_CONSISTENCY, [[1, 5], [3, 15]]),
            # c > 0: x tightens from y with ceil/floor rounding
            ([(-100, 100), (3, 7)], [2], PROP_CONSISTENCY, [[2, 3], [4, 6]]),
            # c < 0: bounds flip
            ([(1, 5), (-100, 100)], [-2], PROP_CONSISTENCY, [[1, 5], [-10, -2]]),
            # c < 0: x tightens from y
            ([(-100, 100), (-10, -2)], [-2], PROP_CONSISTENCY, [[1, 5], [-10, -2]]),
            # c = 0: y forced to 0
            ([(-5, 5), (-3, 3)], [0], PROP_ENTAILMENT, [[-5, 5], [0, 0]]),
            # c = 0: y cannot include 0
            ([(-5, 5), (1, 3)], [0], PROP_INCONSISTENCY, None),
            # already consistent: no change
            ([(2, 3), (4, 6)], [2], PROP_CONSISTENCY, [[2, 3], [4, 6]]),
            # inconsistency: y range outside c * x range
            ([(0, 2), (7, 10)], [3], PROP_INCONSISTENCY, None),
            # x straddling 0 with c > 0
            ([(-3, 4), (-100, 100)], [5], PROP_CONSISTENCY, [[-3, 4], [-15, 20]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_mul_c_eq, domains, parameters, consistency_result, expected_domains)
