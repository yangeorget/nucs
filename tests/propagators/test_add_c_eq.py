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
from nucs.propagators.add_c_eq_propagator import compute_domains_add_c_eq
from tests.propagators.propagator_test import PropagatorTest


class TestAddCEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # x tightens from y
            ([(0, 5), (-10, 10)], [3], PROP_CONSISTENCY, [[0, 5], [3, 8]]),
            # y tightens from x; both sides shrink
            ([(-10, 10), (10, 20)], [5], PROP_CONSISTENCY, [[5, 10], [10, 15]]),
            # negative c
            ([(0, 10), (-100, 100)], [-2], PROP_CONSISTENCY, [[0, 10], [-2, 8]]),
            # already consistent: no change
            ([(2, 3), (5, 6)], [3], PROP_CONSISTENCY, [[2, 3], [5, 6]]),
            # inconsistency: y + c and x cannot meet
            ([(0, 5), (10, 20)], [3], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_add_c_eq, domains, parameters, consistency_result, expected_domains)
