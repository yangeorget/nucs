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

from nucs.constants import PROP_CONSISTENCY
from nucs.propagators.affine_eq_propagator import compute_domains_affine_eq
from tests.propagators.propagator_test import PropagatorTest


class TestAffineEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(1, 10), (1, 10)], [1, 1, 8], PROP_CONSISTENCY, [[1, 7], [1, 7]]),
            ([(5, 10), (5, 10), (5, 10)], [1, 1, 1, 27], PROP_CONSISTENCY, [[7, 10], [7, 10], [7, 10]]),
            ([(-2, -1), (2, 3)], [1, 1, 0], PROP_CONSISTENCY, [[-2, -2], [2, 2]]),
            ([(1, 10), (1, 10)], [1, -3, 0], PROP_CONSISTENCY, [[3, 10], [1, 3]]),
            ([(-14, 11), (-4, 5)], [1, 3, 0], PROP_CONSISTENCY, [[-14, 11], [-3, 4]]),
            (
                [4, 3, 5, 9, 1, 8, 6, 2, 7, 0],
                [200, -1000, 100002, 9900, 100000, 20, 1000, 0, -99010, -1, 0],
                PROP_CONSISTENCY,
                [[4, 4], [3, 3], [5, 5], [9, 9], [1, 1], [8, 8], [6, 6], [2, 2], [7, 7], [0, 0]],
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
        self.assert_compute_domains(
            compute_domains_affine_eq, domains, parameters, consistency_result, expected_domains
        )
