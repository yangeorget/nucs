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
from nucs.propagators.leq_c_reif_propagator import compute_domains_leq_c_reif
from tests.propagators.propagator_test import PropagatorTest


class TestLeqCReif(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # c = 0, b <=> x <= y
            # b is 1: enforce x <= y, bounds tightened
            ([(1, 1), (3, 8), (2, 5)], [0], PROP_CONSISTENCY, [[1, 1], [3, 5], [3, 5]]),
            # b is 1: relation already holds -> entailment
            ([(1, 1), (0, 2), (5, 8)], [0], PROP_ENTAILMENT, [[1, 1], [0, 2], [5, 8]]),
            # b is 1: relation impossible -> inconsistency
            ([(1, 1), (5, 5), (2, 2)], [0], PROP_INCONSISTENCY, None),
            # b is 0: enforce x > y, bounds tightened
            ([(0, 0), (3, 8), (4, 6)], [0], PROP_CONSISTENCY, [[0, 0], [5, 8], [4, 6]]),
            # b is 0: negation already holds -> entailment
            ([(0, 0), (7, 9), (2, 4)], [0], PROP_ENTAILMENT, [[0, 0], [7, 9], [2, 4]]),
            # b is 0: negation impossible -> inconsistency
            ([(0, 0), (2, 2), (5, 5)], [0], PROP_INCONSISTENCY, None),
            # b free: relation always true -> b = 1
            ([(0, 1), (0, 2), (5, 8)], [0], PROP_ENTAILMENT, [[1, 1], [0, 2], [5, 8]]),
            # b free: relation always false -> b = 0
            ([(0, 1), (7, 9), (2, 4)], [0], PROP_ENTAILMENT, [[0, 0], [7, 9], [2, 4]]),
            # b free: undecided -> no change
            ([(0, 1), (3, 6), (4, 7)], [0], PROP_CONSISTENCY, [[0, 1], [3, 6], [4, 7]]),
            # c = -1, b <=> x < y
            # b free: x < y always true -> b = 1
            ([(0, 1), (0, 2), (4, 8)], [-1], PROP_ENTAILMENT, [[1, 1], [0, 2], [4, 8]]),
            # b is 1: enforce x <= y - 1, bounds tightened
            ([(1, 1), (3, 8), (2, 6)], [-1], PROP_CONSISTENCY, [[1, 1], [3, 5], [4, 6]]),
            # b is 0: enforce x >= y, bounds tightened
            ([(0, 0), (3, 8), (4, 6)], [-1], PROP_CONSISTENCY, [[0, 0], [4, 8], [4, 6]]),
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
            compute_domains_leq_c_reif, domains, parameters, consistency_result, expected_domains
        )
