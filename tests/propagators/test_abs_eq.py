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
            # y[MIN] > 0 branch, y narrowed by x
            ([(2, 10), (4, 6)], [], PROP_CONSISTENCY, [[4, 6], [4, 6]]),
            # y[MIN] > 0 branch, x narrowed empty -> inconsistency
            ([(5, 6), (1, 3)], [], PROP_INCONSISTENCY, None),
            # y[MAX] < 0 branch, y narrowed by x
            ([(-10, -2), (4, 6)], [], PROP_CONSISTENCY, [[-6, -4], [4, 6]]),
            # y[MAX] < 0 branch, x narrowed empty -> inconsistency
            ([(-6, -5), (1, 3)], [], PROP_INCONSISTENCY, None),
            # y[MAX] < 0 branch, narrowed both, consistent
            ([(-7, -3), (0, 5)], [], PROP_CONSISTENCY, [[-5, -3], [3, 5]]),
            # straddle 0, x[MIN] < 0 gets clamped to 0
            ([(-3, 3), (-2, 2)], [], PROP_CONSISTENCY, [[-2, 2], [0, 2]]),
            # straddle 0, x[MIN] > max_y -> x empty, inconsistency
            ([(-2, 2), (5, 10)], [], PROP_INCONSISTENCY, None),
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
