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
from nucs.propagators.neq_propagator import compute_domains_neq
from tests.propagators.propagator_test import PropagatorTest


class TestNeq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # disjoint ranges -> entailment, nothing to prune
            ([(0, 2), (3, 5)], [], PROP_ENTAILMENT, [[0, 2], [3, 5]]),
            # x bound to the lower bound of y -> prune y's lower bound, entailment
            ([(3, 3), (3, 8)], [], PROP_ENTAILMENT, [[3, 3], [4, 8]]),
            # x bound to the upper bound of y -> prune y's upper bound, entailment
            ([(8, 8), (3, 8)], [], PROP_ENTAILMENT, [[8, 8], [3, 7]]),
            # y bound to the lower bound of x -> prune x's lower bound, entailment
            ([(3, 8), (3, 3)], [], PROP_ENTAILMENT, [[4, 8], [3, 3]]),
            # x bound to a value strictly inside y -> no hole carving, stay consistent
            ([(5, 5), (0, 10)], [], PROP_CONSISTENCY, [[5, 5], [0, 10]]),
            # both bound to the same value -> inconsistency
            ([(4, 4), (4, 4)], [], PROP_INCONSISTENCY, None),
            # both bound to different values -> entailment
            ([(4, 4), (7, 7)], [], PROP_ENTAILMENT, [[4, 4], [7, 7]]),
            # neither bound, overlapping ranges -> no change, consistent
            ([(0, 5), (3, 8)], [], PROP_CONSISTENCY, [[0, 5], [3, 8]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_neq, domains, parameters, consistency_result, expected_domains)
