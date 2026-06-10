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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.mul_eq_propagator import compute_domains_mul_eq
from tests.propagators.propagator_test import PropagatorTest


class TestMulEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # z tightened to the product hull; x, y already bound-consistent
            ([(2, 3), (4, 5), (0, 100)], [], PROP_CONSISTENCY, [[2, 3], [4, 5], [8, 15]]),
            # x and y both forced from z (2 * 3 = 6) -> entailment
            ([(0, 10), (3, 3), (6, 6)], [], PROP_ENTAILMENT, [[2, 2], [3, 3], [6, 6]]),
            # negative factor: z hull is negative, x and y stay tight
            ([(-3, -2), (4, 5), (-100, 100)], [], PROP_CONSISTENCY, [[-3, -2], [4, 5], [-15, -8]]),
            # 2 * 3 != 7 -> inconsistency
            ([(2, 2), (3, 3), (7, 7)], [], PROP_INCONSISTENCY, None),
            # y straddles 0: x is not tightened, but z and y are
            ([(2, 4), (-1, 2), (-100, 100)], [], PROP_CONSISTENCY, [[2, 4], [-1, 2], [-4, 8]]),
            # x fixed to 0 forces z to 0 and entails the constraint for any y
            ([(0, 0), (5, 9), (-100, 100)], [], PROP_ENTAILMENT, [[0, 0], [5, 9], [0, 0]]),
            # division rounds inward: z in [7, 11], y = 3 -> x in [3, 3] (ceil(7/3)=3, floor(11/3)=3),
            # then z is re-tightened to 3 * 3 = 9 and the constraint is entailed
            ([(0, 10), (3, 3), (7, 11)], [], PROP_ENTAILMENT, [[3, 3], [3, 3], [9, 9]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_mul_eq, domains, parameters, consistency_result, expected_domains)
