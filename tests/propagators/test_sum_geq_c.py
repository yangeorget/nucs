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

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.sum_geq_c_propagator import compute_domains_sum_geq_c
from tests.propagators.propagator_test import PropagatorTest


class TestSumGeqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # already entailed: min sum >= c
            ([(5, 10), (5, 10), (5, 10)], [10], PROP_ENTAILMENT, [[5, 10], [5, 10], [5, 10]]),
            # pruning: each var must rise to keep sum >= c
            ([(0, 5), (0, 5), (0, 5)], [12], PROP_CONSISTENCY, [[2, 5], [2, 5], [2, 5]]),
            # all bound, sum < c
            ([(1, 1), (2, 2), (3, 3)], [10], PROP_INCONSISTENCY, None),
            # one var unbound, pinned by filter
            ([(5, 5), (5, 5), (0, 10)], [15], PROP_ENTAILMENT, [[5, 5], [5, 5], [5, 10]]),
            # infeasible: max sum below c
            ([(0, 2), (0, 2), (0, 2)], [10], PROP_INCONSISTENCY, None),
            # no-op: not yet entailed, no tightening possible
            ([(0, 5), (0, 5), (0, 5)], [5], PROP_CONSISTENCY, [[0, 5], [0, 5], [0, 5]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(
            compute_domains_sum_geq_c, domains, parameters, consistency_result, expected_domains
        )
