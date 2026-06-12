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
from nucs.propagators.strictly_increasing_propagator import compute_domains_strictly_increasing
from tests.propagators.propagator_test import PropagatorTest


class TestStrictlyIncreasing(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # each step adds a strict +1 margin on the mins (forward) and -1 on the maxs (backward)
            ([(0, 9), (0, 9), (0, 9)], [], PROP_CONSISTENCY, [[0, 7], [1, 8], [2, 9]]),
            # three strictly increasing values cannot fit in {0, 1} -> inconsistency
            ([(0, 1), (0, 1), (0, 1)], [], PROP_INCONSISTENCY, None),
            # already strictly separated -> entailed
            ([(0, 2), (3, 5), (6, 9)], [], PROP_ENTAILMENT, [[0, 2], [3, 5], [6, 9]]),
            # adjacent ranges, nothing to prune, not entailed
            ([(0, 8), (1, 9)], [], PROP_CONSISTENCY, [[0, 8], [1, 9]]),
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
            compute_domains_strictly_increasing, domains, parameters, consistency_result, expected_domains
        )
