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
from nucs.propagators.equiv_eq_c_propagator import compute_domains_equiv_eq_c

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from tests.propagators.propagator_test import PropagatorTest


class TestEquivEqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(0, 1), (3, 5)], [4], PROP_CONSISTENCY, [[0, 1], [3, 5]]),
            ([(1, 1), (3, 5)], [4], PROP_ENTAILMENT, [[1, 1], [4, 4]]),
            ([(0, 1), (3, 5)], [6], PROP_ENTAILMENT, [[0, 0], [3, 5]]),
            ([(1, 1), (3, 5)], [6], PROP_INCONSISTENCY, None),
            # b=0, x[MIN]==c -> x[MIN] tightened, then x>c so b=0 entailed
            ([(0, 0), (4, 7)], [4], PROP_ENTAILMENT, [[0, 0], [5, 7]]),
            # b=0, x[MAX]==c -> x[MAX] tightened, then x<c so b=0 entailed
            ([(0, 0), (2, 4)], [4], PROP_ENTAILMENT, [[0, 0], [2, 3]]),
            # b=0, x fixed at c -> inconsistency after x[MIN] += 1
            ([(0, 0), (4, 4)], [4], PROP_INCONSISTENCY, None),
            # b unknown, x fixed exactly at c -> b=1 entailed
            ([(0, 1), (4, 4)], [4], PROP_ENTAILMENT, [[1, 1], [4, 4]]),
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
            compute_domains_equiv_eq_c, domains, parameters, consistency_result, expected_domains
        )
