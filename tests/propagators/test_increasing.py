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
from nucs.propagators.increasing_propagator import compute_domains_increasing
from tests.propagators.propagator_test import PropagatorTest


class TestIncreasing(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # forward sweep lifts the mins, backward sweep lowers the maxs
            ([(2, 5), (0, 3), (0, 10)], [], PROP_CONSISTENCY, [[2, 3], [2, 3], [2, 10]]),
            # x_0 forced above x_1 -> empty domain, inconsistency
            ([(5, 5), (0, 3)], [], PROP_INCONSISTENCY, None),
            # already strictly separated -> entailed
            ([(0, 2), (3, 5), (6, 9)], [], PROP_ENTAILMENT, [[0, 2], [3, 5], [6, 9]]),
            # overlapping ranges, nothing to prune, not entailed
            ([(0, 9), (0, 9)], [], PROP_CONSISTENCY, [[0, 9], [0, 9]]),
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
            compute_domains_increasing, domains, parameters, consistency_result, expected_domains
        )
