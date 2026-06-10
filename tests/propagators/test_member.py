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
from nucs.propagators.member_propagator import compute_domains_member
from tests.propagators.propagator_test import PropagatorTest


class TestMember(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # bounds tightened to the nearest allowed values, holes left in place
            ([(0, 10)], [2, 5, 7], PROP_CONSISTENCY, [[2, 7]]),
            # lower bound snaps up to the first allowed value, which is the only one in range -> entailment
            ([(3, 7)], [1, 5, 9], PROP_ENTAILMENT, [[5, 5]]),
            # upper bound snaps down to the last allowed value
            ([(0, 6)], [1, 3, 8], PROP_CONSISTENCY, [[1, 3]]),
            # range collapses onto a single allowed value -> entailment
            ([(4, 6)], [2, 5, 9], PROP_ENTAILMENT, [[5, 5]]),
            # already bound to an allowed value -> entailment
            ([(5, 5)], [2, 5, 9], PROP_ENTAILMENT, [[5, 5]]),
            # already bound to a forbidden value -> inconsistency
            ([(4, 4)], [2, 5, 9], PROP_INCONSISTENCY, None),
            # no allowed value in range (range sits in a hole) -> inconsistency
            ([(3, 4)], [2, 5, 9], PROP_INCONSISTENCY, None),
            # range entirely below every allowed value -> inconsistency
            ([(0, 1)], [2, 5, 9], PROP_INCONSISTENCY, None),
            # allowed values cover the interval contiguously -> entailment
            ([(2, 4)], [1, 2, 3, 4, 5], PROP_ENTAILMENT, [[2, 4]]),
            # contiguous allowed values, bounds tightened then fully covered -> entailment
            ([(0, 10)], [3, 4, 5], PROP_ENTAILMENT, [[3, 5]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_member, domains, parameters, consistency_result, expected_domains)
