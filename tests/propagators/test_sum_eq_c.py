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
from nucs.propagators.sum_eq_c_propagator import compute_domains_sum_eq_c
from tests.propagators.propagator_test import PropagatorTest


class TestSumEqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # normal pruning: each var squeezed by the residual
            ([(1, 5), (1, 5), (1, 5)], [4], PROP_CONSISTENCY, [[1, 2], [1, 2], [1, 2]]),
            # all bound, sum equals c
            ([(1, 1), (2, 2), (3, 3)], [6], PROP_ENTAILMENT, [[1, 1], [2, 2], [3, 3]]),
            # all bound, sum does not equal c
            ([(1, 1), (2, 2), (3, 3)], [7], PROP_INCONSISTENCY, None),
            # one var unbound: pinned to c - rest
            ([(2, 2), (3, 3), (0, 10)], [9], PROP_ENTAILMENT, [[2, 2], [3, 3], [4, 4]]),
            # infeasible: max sum below c
            ([(1, 2), (1, 2), (1, 2)], [10], PROP_INCONSISTENCY, None),
            # infeasible: min sum above c
            ([(5, 9), (5, 9), (5, 9)], [3], PROP_INCONSISTENCY, None),
            # already tight, no pruning possible
            ([(1, 5), (1, 5), (1, 5)], [9], PROP_CONSISTENCY, [[1, 5], [1, 5], [1, 5]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_sum_eq_c, domains, parameters, consistency_result, expected_domains)
