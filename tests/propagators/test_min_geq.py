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
from nucs.propagators.min_geq_propagator import compute_domains_min_geq
from tests.propagators.propagator_test import PropagatorTest


class TestMinGeq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(1, 4), (2, 5), (2, 6)], [], PROP_CONSISTENCY, [[2, 4], [2, 5], [2, 4]]),
            ([(1, 3), (3, 3), (4, 5)], [], PROP_INCONSISTENCY, None),
            ([(2, 4), (2, 5), (6, 8)], [], PROP_INCONSISTENCY, None),
            ([(2, 3), (2, 3), (0, 1)], [], PROP_ENTAILMENT, [[2, 3], [2, 3], [0, 1]]),
            ([(0, 1), (0, 1), (1, 1)], [], PROP_CONSISTENCY, [[1, 1], [1, 1], [1, 1]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_min_geq, domains, parameters, consistency_result, expected_domains)
