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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from typing import List, Optional, Tuple, Union

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.count_eq_propagator import compute_domains_count_eq
from tests.propagators.propagator_test import PropagatorTest


class TestCountEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 1],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5], [1, 1]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 2],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 0],
                [5],
                PROP_INCONSISTENCY,
                [],
            ),
            (
                [(1, 4), 5, (3, 6), (6, 8), 3, 5, (1, 2)],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [5, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, (-1, 10)],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [1, 3]],
            ),
            (
                [2, (0, 1), (3, 4), 2, 2, (2, 4)],
                [2],
                PROP_ENTAILMENT,
                [[2, 2], [0, 1], [3, 4], [2, 2], [2, 2], [3, 3]],
            ),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_count_eq, domains, parameters, consistency_result, expected_domains)
