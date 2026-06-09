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
from nucs.propagators.linear_eq_c_propagator import compute_domains_linear_eq_c
from tests.propagators.propagator_test import PropagatorTest


class TestLinearEqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # negative coefficient pruning: 2x - 3y = 0
            ([(1, 10), (1, 10)], [2, -3, 0], PROP_CONSISTENCY, [[3, 9], [2, 6]]),
            # positive coefficients pruning
            ([(1, 10), (1, 10)], [1, 1, 8], PROP_CONSISTENCY, [[1, 7], [1, 7]]),
            # three variables, all factor 1
            ([(5, 10), (5, 10), (5, 10)], [1, 1, 1, 27], PROP_CONSISTENCY, [[7, 10], [7, 10], [7, 10]]),
            # negative c parameter, both bounds collapse
            ([(-2, -1), (2, 3)], [1, 1, 0], PROP_ENTAILMENT, [[-2, -2], [2, 2]]),
            # all already bound, equation holds
            (
                [4, 3, 5, 9, 1, 8, 6, 2, 7, 0],
                [200, -1000, 100002, 9900, 100000, 20, 1000, 0, -99010, -1, 0],
                PROP_ENTAILMENT,
                [[4, 4], [3, 3], [5, 5], [9, 9], [1, 1], [8, 8], [6, 6], [2, 2], [7, 7], [0, 0]],
            ),
            # inconsistency: max reachable sum < c
            ([(1, 2), (1, 2)], [1, 1, 5], PROP_INCONSISTENCY, None),
            # inconsistency: min reachable sum > c
            ([(5, 9), (5, 9)], [1, 1, 5], PROP_INCONSISTENCY, None),
            # inconsistency with negative coefficient
            ([(0, 1), (5, 6)], [1, -1, 0], PROP_INCONSISTENCY, None),
            # zero coefficient: x_1 ignored
            ([(1, 10), (1, 10), (1, 10)], [1, 0, 1, 8], PROP_CONSISTENCY, [[1, 7], [1, 10], [1, 7]]),
            # all bound, equation holds: entailment
            ([(2, 2), (3, 3)], [1, 1, 5], PROP_ENTAILMENT, [[2, 2], [3, 3]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(
            compute_domains_linear_eq_c, domains, parameters, consistency_result, expected_domains
        )
