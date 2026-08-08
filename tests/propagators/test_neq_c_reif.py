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

import numpy as np
import pytest

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.neq_c_reif_propagator import compute_domains_neq_c_reif
from tests.propagators.propagator_test import PropagatorTest


class TestNeqCReif(PropagatorTest):
    @pytest.mark.parametrize("c", [-1, 0, 1, 2, 3])
    def test_bound_consistency_against_brute_force(self, c: int) -> None:
        # the reified propagator must compute the exact bound-consistent projection of b <=> (x != c)
        for b_dom in [(0, 0), (1, 1), (0, 1)]:
            for xl in range(-2, 5):
                for xu in range(xl, 5):
                    feasible = [
                        (bv, xv)
                        for bv in range(b_dom[0], b_dom[1] + 1)
                        for xv in range(xl, xu + 1)
                        if bv == (1 if xv != c else 0)
                    ]
                    domains = np.array([list(b_dom), [xl, xu]], dtype=np.int32)
                    status = compute_domains_neq_c_reif(domains, np.array([c], dtype=np.int32))
                    if not feasible:
                        assert status == PROP_INCONSISTENCY, f"expected inconsistency b={b_dom} x={xl}..{xu} c={c}"
                        continue
                    assert status in (PROP_CONSISTENCY, PROP_ENTAILMENT)
                    bs = [t[0] for t in feasible]
                    xs = [t[1] for t in feasible]
                    assert domains[0, MIN] == min(bs) and domains[0, MAX] == max(bs), f"b for {xl}..{xu} c={c}"
                    assert domains[1, MIN] == min(xs) and domains[1, MAX] == max(xs), f"x for {xl}..{xu} c={c}"

    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # b=1 (x != c): drop c from the bound; x no longer contains c, so x != c becomes entailed
            ([(1, 1), (2, 9)], [2], PROP_ENTAILMENT, [[1, 1], [3, 9]]),
            # b=1, c interior: bounds cannot remove it
            ([(1, 1), (0, 9)], [4], PROP_CONSISTENCY, [[1, 1], [0, 9]]),
            # b=0 (x == c): fix x to c
            ([(0, 0), (0, 9)], [4], PROP_ENTAILMENT, [[0, 0], [4, 4]]),
            # b=0 but c not in x -> inconsistent
            ([(0, 0), (5, 9)], [4], PROP_INCONSISTENCY, None),
            # b free, x does not contain c -> b = 1
            ([(0, 1), (5, 9)], [4], PROP_ENTAILMENT, [[1, 1], [5, 9]]),
            # b free, x fixed to c -> b = 0
            ([(0, 1), (4, 4)], [4], PROP_ENTAILMENT, [[0, 0], [4, 4]]),
            # b free, x straddles c -> undecided
            ([(0, 1), (0, 9)], [4], PROP_CONSISTENCY, [[0, 1], [0, 9]]),
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
            compute_domains_neq_c_reif, domains, parameters, consistency_result, expected_domains
        )
