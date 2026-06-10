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
from nucs.propagators.linear_neq_c_propagator import compute_domains_linear_neq_c
from tests.propagators.propagator_test import PropagatorTest


class TestLinearNeqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # two unbound variables -> nothing can be filtered, stay consistent
            ([(0, 5), (0, 5)], [1, 1, 5], PROP_CONSISTENCY, [[0, 5], [0, 5]]),
            # one unbound, forbidden value on its lower bound -> prune it, entailment
            ([(2, 2), (3, 8)], [1, 1, 5], PROP_ENTAILMENT, [[2, 2], [4, 8]]),
            # one unbound, forbidden value on its upper bound -> prune it, entailment
            ([(2, 2), (0, 3)], [1, 1, 5], PROP_ENTAILMENT, [[2, 2], [0, 2]]),
            # one unbound, forbidden value strictly inside -> no hole carving, stay consistent
            ([(2, 2), (0, 10)], [1, 1, 5], PROP_CONSISTENCY, [[2, 2], [0, 10]]),
            # one unbound, the residual is not divisible by the factor -> can never be equal, entailment
            ([(3, 3), (0, 10)], [1, 2, 6], PROP_ENTAILMENT, [[3, 3], [0, 10]]),
            # negative coefficient: 2x - 3y != 0, x bound to 3 -> y != 2, prune lower bound
            ([(3, 3), (2, 10)], [2, -3, 0], PROP_ENTAILMENT, [[3, 3], [3, 10]]),
            # all bound and the sum equals c -> inconsistency
            ([(2, 2), (3, 3)], [1, 1, 5], PROP_INCONSISTENCY, None),
            # all bound and the sum differs from c -> entailment
            ([(2, 2), (3, 3)], [1, 1, 6], PROP_ENTAILMENT, [[2, 2], [3, 3]]),
            # c outside the reachable range -> entailment regardless of unbound count
            ([(0, 1), (0, 1)], [1, 1, 5], PROP_ENTAILMENT, [[0, 1], [0, 1]]),
            # zero coefficient: the second variable is ignored
            ([(2, 2), (0, 10), (3, 8)], [1, 0, 1, 5], PROP_ENTAILMENT, [[2, 2], [0, 10], [4, 8]]),
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
            compute_domains_linear_neq_c, domains, parameters, consistency_result, expected_domains
        )
