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
from nucs.propagators.eq_propagator import compute_domains_eq
from tests.propagators.propagator_test import PropagatorTest


class TestEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # overlapping ranges are intersected on both variables
            ([(0, 5), (3, 8)], [], PROP_CONSISTENCY, [[3, 5], [3, 5]]),
            # already tight and equal -> no change, still consistent
            ([(2, 7), (2, 7)], [], PROP_CONSISTENCY, [[2, 7], [2, 7]]),
            # intersection reduces to a single value -> entailment
            ([(0, 3), (3, 9)], [], PROP_ENTAILMENT, [[3, 3], [3, 3]]),
            # both already bound to the same value -> entailment
            ([(4, 4), (4, 4)], [], PROP_ENTAILMENT, [[4, 4], [4, 4]]),
            # disjoint ranges -> inconsistency
            ([(0, 2), (3, 5)], [], PROP_INCONSISTENCY, None),
            # one bound variable forces the other
            ([(5, 5), (0, 10)], [], PROP_ENTAILMENT, [[5, 5], [5, 5]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_eq, domains, parameters, consistency_result, expected_domains)
