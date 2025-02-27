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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT
from nucs.propagators.affine_geq_propagator import compute_domains_affine_geq
from tests.propagators.propagator_test import PropagatorTest


class TestAffineGeq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(1, 10), (1, 10)], [1, -1, 1], PROP_CONSISTENCY, [[2, 10], [1, 9]]),
            ([(5, 10), (5, 10), (5, 10)], [1, 1, 1, 27], PROP_CONSISTENCY, [[7, 10], [7, 10], [7, 10]]),
            ([(5, 10), (1, 2)], [1, 1, 6], PROP_ENTAILMENT, [[5, 10], [1, 2]]),
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
            compute_domains_affine_geq, domains, parameters, consistency_result, expected_domains
        )
