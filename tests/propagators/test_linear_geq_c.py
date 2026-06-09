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
from nucs.propagators.linear_geq_c_propagator import compute_domains_linear_geq_c
from tests.propagators.propagator_test import PropagatorTest


class TestLinearGeqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # mixed-sign coefficients: x - y >= 1
            ([(1, 10), (1, 10)], [1, -1, 1], PROP_CONSISTENCY, [[2, 10], [1, 9]]),
            # positive coefficients pruning
            ([(5, 10), (5, 10), (5, 10)], [1, 1, 1, 27], PROP_CONSISTENCY, [[7, 10], [7, 10], [7, 10]]),
            # already entailed: min sum >= c
            ([(5, 10), (1, 2)], [1, 1, 6], PROP_ENTAILMENT, [[5, 10], [1, 2]]),
            # negative coefficient pruning: -x + y >= 5
            ([(0, 10), (0, 10)], [-1, 1, 5], PROP_CONSISTENCY, [[0, 5], [5, 10]]),
            # zero coefficient: x_1 ignored
            ([(0, 5), (0, 5), (0, 5)], [1, 0, 1, 8], PROP_CONSISTENCY, [[3, 5], [0, 5], [3, 5]]),
            # inconsistency: max reachable sum < c
            ([(0, 2), (0, 2)], [1, 1, 10], PROP_INCONSISTENCY, None),
            # inconsistency with negative coefficient: -x >= 1 with x in [2, 5]
            ([(2, 5)], [-1, 1], PROP_INCONSISTENCY, None),
            # no-op: not entailed, no tightening possible
            ([(0, 5), (0, 5)], [1, 1, 5], PROP_CONSISTENCY, [[0, 5], [0, 5]]),
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
            compute_domains_linear_geq_c, domains, parameters, consistency_result, expected_domains
        )
