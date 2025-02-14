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
from nucs.propagators.count_leq_c_propagator import compute_domains_count_leq_c
from tests.propagators.propagator_test import PropagatorTest


class TestCountLeqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5],
                [5, 0],
                PROP_INCONSISTENCY,
                None,
            ),
            (
                [(1, 4), (3, 5), (3, 5), (6, 8), 3, 5],
                [5, 1],
                PROP_ENTAILMENT,
                [[1, 4], [3, 4], [3, 4], [6, 8], [3, 3], [5, 5]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5],
                [5, 1],
                PROP_CONSISTENCY,
                [[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5]],
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
            compute_domains_count_leq_c, domains, parameters, consistency_result, expected_domains
        )
