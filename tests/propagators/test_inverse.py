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

import random

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.propagators.inverse_propagator import compute_domains_inverse
from tests.propagators.propagator_test import PropagatorTest, random_bounds


class TestInverse(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # all bound, valid permutation -> consistency, unchanged
            ([1, 0, 1, 0], [], PROP_CONSISTENCY, [[1, 1], [0, 0], [1, 1], [0, 0]]),
            # prev all bound, propagates to next (prev[0]=1 -> next[1]=0; prev[1]=0 -> next[0]=1)
            ([(0, 1), (0, 1), 1, 0], [], PROP_CONSISTENCY, [[1, 1], [0, 0], [1, 1], [0, 0]]),
            # next all bound, propagates to prev (by symmetry)
            ([1, 0, (0, 1), (0, 1)], [], PROP_CONSISTENCY, [[1, 1], [0, 0], [1, 1], [0, 0]]),
            # both next pointing to 0 -> inconsistent
            ([0, 0, (0, 1), (0, 1)], [], PROP_INCONSISTENCY, None),
            # partial filtering: next[0]=0 forces prev[0]=0 and prev[1]=1
            ([0, (0, 1), (0, 1), (0, 1)], [], PROP_CONSISTENCY, [[0, 0], [1, 1], [0, 0], [1, 1]]),
            # disjoint prefix raises prev[j,DOMAIN_MIN]; disjoint suffix lowers prev[j,DOMAIN_MAX]
            (
                [2, 1, 0, (0, 2), (0, 2), (0, 2)],
                [],
                PROP_CONSISTENCY,
                [[2, 2], [1, 1], [0, 0], [2, 2], [1, 1], [0, 0]],
            ),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_inverse, domains, parameters, consistency_result, expected_domains)

    @pytest.mark.parametrize("offsets", [[], [0, 0], [1, 1], [2, 0], [0, 2]])
    def test_soundness_against_brute_force(self, offsets: list[int]) -> None:
        # next[i] = next_values + j iff prev[j] = prev_values + i, with values one past each end of the labels
        next_values, prev_values = offsets or [0, 0]
        rng = random.Random(20260918 + len(offsets) + 3 * next_values + prev_values)
        for _ in range(400):
            n = rng.randint(1, 3)
            bounds = random_bounds(rng, n, next_values - 1, next_values + n) + random_bounds(
                rng, n, prev_values - 1, prev_values + n
            )

            def is_solution(point: tuple[int, ...], n: int = n) -> bool:
                succ, pred = point[:n], point[n:]
                return all(
                    0 <= succ[i] - next_values < n and pred[succ[i] - next_values] == prev_values + i for i in range(n)
                ) and all(
                    0 <= pred[j] - prev_values < n and succ[pred[j] - prev_values] == next_values + j for j in range(n)
                )

            self.assert_sound_against_brute_force(compute_domains_inverse, bounds, offsets, is_solution)
