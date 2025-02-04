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
from nucs.propagators.element_l_eq_propagator import compute_domains_element_l_eq
from tests.propagators.propagator_test import PropagatorTest


class TestElementLEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(-1, 0), (1, 2), (0, 2), (-1, 1)], [], PROP_CONSISTENCY, [[-1, 0], [1, 2], [0, 1], [-1, 1]]),
            ([(-4, -2), (1, 2), (0, 1), (0, 1)], [], PROP_ENTAILMENT, [[-4, -2], [1, 1], [1, 1], [1, 1]]),
            (
                [3, 0, 1, 2, 4, (0, 4), (-2, 0)],
                [0],
                PROP_ENTAILMENT,
                [[3, 3], [0, 0], [1, 1], [2, 2], [4, 4], [1, 1], [0, 0]],
            ),
            ([(-4, -2), (1, 2), (0, 1), (0, 0)], [], PROP_INCONSISTENCY, None),
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
            compute_domains_element_l_eq, domains, parameters, consistency_result, expected_domains
        )
