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
from nucs.propagators.min_eq_propagator import compute_domains_min_eq
from tests.propagators.propagator_test import PropagatorTest


class TestMinEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(1, 4), (2, 5), (2, 6)], [], PROP_CONSISTENCY, [[2, 4], [2, 5], [2, 4]]),
            ([(1, 3), (3, 3), (4, 5)], [], PROP_INCONSISTENCY, None),
            ([(2, 4), (2, 5), (6, 8)], [], PROP_INCONSISTENCY, None),
            ([(0, 1), (0, 1), (1, 1)], [], PROP_CONSISTENCY, [[1, 1], [1, 1], [1, 1]]),
            ([(2, 3), (2, 3), (0, 1)], [], PROP_INCONSISTENCY, None),
            # another x_i can reach y_max, so x_0 is not the unique minimizer and must not be forced down
            ([(0, 10), (5, 5), (0, 10)], [], PROP_CONSISTENCY, [[0, 10], [5, 5], [0, 5]]),
            # x_0 cannot reach y_max, so x_1 is the unique minimizer and is forced to at most y_max
            ([(7, 10), (0, 4), (0, 3)], [], PROP_CONSISTENCY, [[7, 10], [0, 3], [0, 3]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_min_eq, domains, parameters, consistency_result, expected_domains)
