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
from nucs.propagators.sum_eq_propagator import compute_domains_sum_eq
from tests.propagators.propagator_test import PropagatorTest


class TestSumEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,consistency_result,expected_domains",
        [
            # normal pruning: last var is the result
            ([(1, 10), (1, 10), (1, 10)], PROP_CONSISTENCY, [[1, 9], [1, 9], [2, 10]]),
            # all bound, sum matches result
            ([(2, 2), (3, 3), (5, 5)], PROP_ENTAILMENT, [[2, 2], [3, 3], [5, 5]]),
            # all bound, sum does not match
            ([(2, 2), (3, 3), (6, 6)], PROP_INCONSISTENCY, None),
            # only the result var unbound: pinned to exact sum
            ([(2, 2), (3, 3), (0, 10)], PROP_ENTAILMENT, [[2, 2], [3, 3], [5, 5]]),
            # only one sum var unbound: pinned to residual
            ([(2, 2), (0, 10), (5, 5)], PROP_ENTAILMENT, [[2, 2], [3, 3], [5, 5]]),
            # infeasible: max sum below result
            ([(1, 2), (1, 2), (10, 10)], PROP_INCONSISTENCY, None),
            # infeasible: min sum above result
            ([(5, 6), (5, 6), (1, 1)], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_sum_eq, domains, [], consistency_result, expected_domains)
