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
from nucs.propagators.abs_eq_propagator import compute_domains_abs_eq
from tests.propagators.propagator_test import PropagatorTest


class TestAbsEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(-4, 4), (-5, 5)], [], PROP_CONSISTENCY, [[-4, 4], [0, 4]]),
            ([(-4, 4), (-5, 0)], [], PROP_CONSISTENCY, [[0, 0], [0, 0]]),
            ([(2, 4), (-5, 5)], [], PROP_CONSISTENCY, [[2, 4], [2, 4]]),
            ([(-4, -2), (-5, 5)], [], PROP_CONSISTENCY, [[-4, -2], [2, 4]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_abs_eq, domains, parameters, consistency_result, expected_domains)
