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
from nucs.propagators.sum_eq_propagator import compute_domains_sum_eq
from tests.propagators.propagator_test import PropagatorTest


class TestSumEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,consistency_result,expected_domains",
        [
            ([(1, 10), (1, 10), (1, 10)], PROP_CONSISTENCY, [[1, 9], [1, 9], [2, 10]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_sum_eq, domains, [], consistency_result, expected_domains)
