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

from nucs.constants import PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.propagators.max_eq_propagator import compute_domains_max_eq
from tests.propagators.propagator_test import PropagatorTest


class TestMaxEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(1, 4), (2, 5), (0, 2)], [], PROP_CONSISTENCY, [[1, 2], [2, 2], [2, 2]]),
            ([(1, 4), (3, 5), (0, 2)], [], PROP_INCONSISTENCY, None),
            ([(2, 4), (2, 5), (0, 1)], [], PROP_INCONSISTENCY, None),
            ([(0, 1), (0, 1), (0, 0)], [], PROP_CONSISTENCY, [[0, 0], [0, 0], [0, 0]]),
            ([(0, 1), (0, 1), (2, 3)], [], PROP_INCONSISTENCY, None),
            # another x_i can reach y_min, so x_0 is not the unique maximizer and must not be forced up
            ([(0, 10), (5, 5), (0, 10)], [], PROP_CONSISTENCY, [[0, 10], [5, 5], [5, 10]]),
            # x_0 cannot reach y_min, so x_1 is the unique maximizer and is forced to at least y_min
            ([(0, 3), (6, 10), (7, 20)], [], PROP_CONSISTENCY, [[0, 3], [7, 10], [7, 10]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_max_eq, domains, parameters, consistency_result, expected_domains)
