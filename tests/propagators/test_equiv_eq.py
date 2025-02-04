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
from nucs.propagators.equiv_eq_propagator import compute_domains_equiv_eq
from tests.propagators.propagator_test import PropagatorTest


class TestEquivEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(0, 1), (3, 5)], [4], PROP_CONSISTENCY, [[0, 1], [3, 5]]),
            ([(1, 1), (3, 5)], [4], PROP_ENTAILMENT, [[1, 1], [4, 4]]),
            ([(0, 1), (3, 5)], [6], PROP_ENTAILMENT, [[0, 0], [3, 5]]),
            ([(1, 1), (3, 5)], [6], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_equiv_eq, domains, parameters, consistency_result, expected_domains)
