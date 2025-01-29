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
from nucs.propagators.gcc_propagator import compute_domains_gcc
from tests.propagators.propagator_test import PropagatorTest


class TestGCC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([0], [0, 1, 1], PROP_CONSISTENCY, [[0, 0]]),
            ([0, 1], [0, 1, 1, 1, 1], PROP_CONSISTENCY, [[0, 0], [1, 1]]),
            ([0, (0, 1)], [0, 1, 1, 1, 1], PROP_CONSISTENCY, [[0, 0], [1, 1]]),
            ([0, 2, (1, 2)], [0] + [1] * 6, PROP_CONSISTENCY, [[0, 0], [2, 2], [1, 1]]),
            (
                [0, (0, 4), (0, 4), (0, 4), (0, 4)],
                [0] + [1] * 10,
                PROP_CONSISTENCY,
                [[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]],
            ),
            (
                [(3, 6), (3, 4), (2, 5), (2, 4), (3, 4), (1, 6)],
                [1] + [1] * 12,
                PROP_CONSISTENCY,
                [[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]],
            ),
            (
                [(3, 4), (2, 4), (3, 4), (2, 5), (3, 6), (1, 6)],
                [1] + [0] * 6 + [1] * 6,
                PROP_CONSISTENCY,
                [[3, 4], [2, 2], [3, 4], [5, 5], [6, 6], [1, 1]],
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
        self.assert_compute_domains(compute_domains_gcc, domains, parameters, consistency_result, expected_domains)
