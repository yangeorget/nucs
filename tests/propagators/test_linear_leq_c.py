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
from nucs.propagators.linear_leq_c_propagator import compute_domains_linear_leq_c
from tests.propagators.propagator_test import PropagatorTest


class TestLinearLeqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # mixed-sign coefficients: x - y <= -1
            ([(1, 10), (1, 10)], [1, -1, -1], PROP_CONSISTENCY, [[1, 9], [2, 10]]),
            # positive coefficients pruning
            ([(1, 10), (1, 10)], [1, 1, 8], PROP_CONSISTENCY, [[1, 7], [1, 7]]),
            # already entailed: max sum <= c
            ([(2, 3), (1, 2)], [1, 1, 5], PROP_ENTAILMENT, [[2, 3], [1, 2]]),
            # negative coefficient pruning: -x + y <= -3
            ([(0, 10), (0, 10)], [-1, 1, -3], PROP_CONSISTENCY, [[3, 10], [0, 7]]),
            # zero coefficient: x_1 ignored
            ([(0, 5), (0, 5), (0, 5)], [1, 0, 1, 3], PROP_CONSISTENCY, [[0, 3], [0, 5], [0, 3]]),
            # inconsistency: min reachable sum > c
            ([(5, 9), (5, 9)], [1, 1, 5], PROP_INCONSISTENCY, None),
            # inconsistency with negative coefficient: -x <= -5 with x in [0, 2]
            ([(0, 2)], [-1, -5], PROP_INCONSISTENCY, None),
            # no-op: not entailed, no tightening possible
            ([(0, 3), (0, 3)], [1, 1, 4], PROP_CONSISTENCY, [[0, 3], [0, 3]]),
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
            compute_domains_linear_leq_c, domains, parameters, consistency_result, expected_domains
        )
