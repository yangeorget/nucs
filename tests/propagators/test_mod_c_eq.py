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
from nucs.propagators.mod_c_eq_propagator import compute_domains_mod_c_eq
from tests.propagators.propagator_test import PropagatorTest


def trunc_mod(x: int, m: int) -> int:
    """The truncated remainder x mod m (sign of the dividend), independent of the sign of m."""
    am = abs(m)
    return x % am if x >= 0 else -((-x) % am)


class TestModCEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # z tightened to the image of x mod m, here crossing a multiple of 5
            ([(0, 9), (-10, 10)], [5], PROP_CONSISTENCY, [[0, 9], [0, 4]]),
            # x stays within one block of m, so z is the contiguous [xl mod m, xu mod m]
            ([(10, 12), (-10, 10)], [5], PROP_CONSISTENCY, [[10, 12], [0, 2]]),
            # x pruned to the residue class fixed by z
            ([(0, 100), (3, 3)], [7], PROP_CONSISTENCY, [[3, 94], [3, 3]]),
            # truncated remainder takes the sign of the dividend
            ([(-7, -7), (-10, 10)], [5], PROP_ENTAILMENT, [[-7, -7], [-2, -2]]),
            # the modulus sign is irrelevant: 17 mod -5 = 2
            ([(17, 17), (-10, 10)], [-5], PROP_ENTAILMENT, [[17, 17], [2, 2]]),
            # x mod 1 is always 0
            ([(5, 9), (-3, 3)], [1], PROP_CONSISTENCY, [[5, 9], [0, 0]]),
            # no x in [0, 2] has a remainder in [3, 4]
            ([(0, 2), (3, 4)], [5], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_mod_c_eq, domains, parameters, consistency_result, expected_domains)

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, -3, -4, -7])
    def test_bound_consistency_against_brute_force(self, m: int) -> None:
        # exhaustively check that the propagator computes the exact bound-consistent projection
        for xl in range(-6, 7):
            for xu in range(xl, 7):
                for zl in range(-5, 6):
                    for zu in range(zl, 6):
                        feasible = [(xv, trunc_mod(xv, m)) for xv in range(xl, xu + 1) if zl <= trunc_mod(xv, m) <= zu]
                        domains = np.array([[xl, xu], [zl, zu]], dtype=np.int32)
                        status = compute_domains_mod_c_eq(domains, np.array([m], dtype=np.int32))
                        if not feasible:
                            assert status == PROP_INCONSISTENCY, (
                                f"expected inconsistency for {xl}..{xu} {zl}..{zu} m={m}"
                            )
                            continue
                        assert status in (PROP_CONSISTENCY, PROP_ENTAILMENT)
                        xs = [p[0] for p in feasible]
                        zs = [p[1] for p in feasible]
                        assert domains[0, MIN] == min(xs) and domains[0, MAX] == max(xs), f"x for {xl}..{xu} m={m}"
                        assert domains[1, MIN] == min(zs) and domains[1, MAX] == max(zs), f"z for {xl}..{xu} m={m}"
