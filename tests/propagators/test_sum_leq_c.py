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
from nucs.propagators.sum_leq_c_propagator import compute_domains_sum_leq_c
from tests.propagators.propagator_test import PropagatorTest


class TestSumLeqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # already entailed: max sum <= c
            ([(0, 3), (0, 3), (0, 3)], [10], PROP_ENTAILMENT, [[0, 3], [0, 3], [0, 3]]),
            # pruning: each var must fall to keep sum <= c
            ([(0, 5), (0, 5), (0, 5)], [3], PROP_CONSISTENCY, [[0, 3], [0, 3], [0, 3]]),
            # all bound, sum > c
            ([(5, 5), (5, 5), (5, 5)], [10], PROP_INCONSISTENCY, None),
            # one var unbound, pinned by filter
            ([(2, 2), (3, 3), (0, 10)], [10], PROP_ENTAILMENT, [[2, 2], [3, 3], [0, 5]]),
            # infeasible: min sum above c
            ([(5, 9), (5, 9), (5, 9)], [10], PROP_INCONSISTENCY, None),
            # no-op: not yet entailed, no tightening possible
            ([(0, 5), (0, 5), (0, 5)], [10], PROP_CONSISTENCY, [[0, 5], [0, 5], [0, 5]]),
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
            compute_domains_sum_leq_c, domains, parameters, consistency_result, expected_domains
        )
