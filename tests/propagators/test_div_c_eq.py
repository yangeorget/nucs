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

import numpy as np
import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.div_c_eq_propagator import compute_domains_div_c_eq
from tests.propagators.propagator_test import PropagatorTest


def trunc_div(x: int, c: int) -> int:
    """The truncated quotient x div c (rounded toward zero)."""
    q = abs(x) // abs(c)
    return q if (x >= 0) == (c >= 0) else -q


class TestDivCEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # y tightened to the image of x div c
            ([(0, 9), (-10, 10)], [5], PROP_CONSISTENCY, [[0, 9], [0, 1]]),
            # x pruned to the preimage of a fixed quotient
            ([(0, 100), (3, 3)], [7], PROP_CONSISTENCY, [[21, 27], [3, 3]]),
            # truncated division rounds toward zero, so -7 div 5 = -1
            ([(-7, -7), (-10, 10)], [5], PROP_ENTAILMENT, [[-7, -7], [-1, -1]]),
            # a negative divisor flips the sign: 17 div -5 = -3
            ([(17, 17), (-10, 10)], [-5], PROP_ENTAILMENT, [[17, 17], [-3, -3]]),
            # negative divisor over a range: x in [0, 10] maps to [-3, 0]
            ([(0, 10), (-10, 10)], [-3], PROP_CONSISTENCY, [[0, 10], [-3, 0]]),
            # x div 1 = x
            ([(5, 9), (-20, 20)], [1], PROP_CONSISTENCY, [[5, 9], [5, 9]]),
            # preimage of a negative quotient block, clipped to x's domain
            ([(-9, -1), (-4, -4)], [2], PROP_CONSISTENCY, [[-9, -8], [-4, -4]]),
            # no x in [0, 2] has a quotient in [3, 4] when dividing by 5
            ([(0, 2), (3, 4)], [5], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_div_c_eq, domains, parameters, consistency_result, expected_domains)

    @pytest.mark.parametrize("c", [1, 2, 3, 4, 5, 7, -2, -3, -5])
    def test_bound_consistency_against_brute_force(self, c: int) -> None:
        # exhaustively check that the propagator computes the exact bound-consistent projection
        for xl in range(-8, 9):
            for xu in range(xl, 9):
                for yl in range(-5, 6):
                    for yu in range(yl, 6):
                        feasible = [(xv, trunc_div(xv, c)) for xv in range(xl, xu + 1) if yl <= trunc_div(xv, c) <= yu]
                        domains = np.array([[xl, xu], [yl, yu]], dtype=np.int32)
                        status = compute_domains_div_c_eq(domains, np.array([c], dtype=np.int32))
                        if not feasible:
                            assert status == PROP_INCONSISTENCY, (
                                f"expected inconsistency for {xl}..{xu} {yl}..{yu} c={c}"
                            )
                            continue
                        assert status in (PROP_CONSISTENCY, PROP_ENTAILMENT)
                        xs = [p[0] for p in feasible]
                        ys = [p[1] for p in feasible]
                        assert domains[0, DOMAIN_MIN] == min(xs) and domains[0, DOMAIN_MAX] == max(xs), (
                            f"x for {xl}..{xu} c={c}"
                        )
                        assert domains[1, DOMAIN_MIN] == min(ys) and domains[1, DOMAIN_MAX] == max(ys), (
                            f"y for {xl}..{xu} c={c}"
                        )
