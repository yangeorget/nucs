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

import numpy as np
import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.linear_eq_c_propagator import compute_domains_linear_eq_c
from tests.propagators.propagator_test import PropagatorTest


class TestLinearEqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # negative coefficient pruning: 2x - 3y = 0
            ([(1, 10), (1, 10)], [2, -3, 0], PROP_CONSISTENCY, [[3, 9], [2, 6]]),
            # positive coefficients pruning
            ([(1, 10), (1, 10)], [1, 1, 8], PROP_CONSISTENCY, [[1, 7], [1, 7]]),
            # three variables, all factor 1
            ([(5, 10), (5, 10), (5, 10)], [1, 1, 1, 27], PROP_CONSISTENCY, [[7, 10], [7, 10], [7, 10]]),
            # negative c parameter, both bounds collapse
            ([(-2, -1), (2, 3)], [1, 1, 0], PROP_ENTAILMENT, [[-2, -2], [2, 2]]),
            # all already bound, equation holds
            (
                [4, 3, 5, 9, 1, 8, 6, 2, 7, 0],
                [200, -1000, 100002, 9900, 100000, 20, 1000, 0, -99010, -1, 0],
                PROP_ENTAILMENT,
                [[4, 4], [3, 3], [5, 5], [9, 9], [1, 1], [8, 8], [6, 6], [2, 2], [7, 7], [0, 0]],
            ),
            # inconsistency: max reachable sum < c
            ([(1, 2), (1, 2)], [1, 1, 5], PROP_INCONSISTENCY, None),
            # inconsistency: min reachable sum > c
            ([(5, 9), (5, 9)], [1, 1, 5], PROP_INCONSISTENCY, None),
            # inconsistency with negative coefficient
            ([(0, 1), (5, 6)], [1, -1, 0], PROP_INCONSISTENCY, None),
            # zero coefficient: x_1 ignored
            ([(1, 10), (1, 10), (1, 10)], [1, 0, 1, 8], PROP_CONSISTENCY, [[1, 7], [1, 10], [1, 7]]),
            # all bound, equation holds: entailment
            ([(2, 2), (3, 3)], [1, 1, 5], PROP_ENTAILMENT, [[2, 2], [3, 3]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(
            compute_domains_linear_eq_c, domains, parameters, consistency_result, expected_domains
        )

    def test_same_fixpoint_as_the_one_pass_rule(self) -> None:
        # the in-call fixpoint narrows exactly as far as the one-pass rule the propagator used to apply, repeated
        # until nothing changes -- so switching between them cannot change a search tree
        rng = random.Random(20260917)
        for _ in range(5000):
            n = rng.randint(1, 5)
            factors = [rng.choice([-3, -2, -1, 0, 1, 2, 3]) for _ in range(n)]
            bounds = []
            for _ in range(n):
                lo = rng.randint(-6, 6)
                bounds.append((lo, lo + rng.randint(0, 8)))
            c = rng.randint(-15, 15)
            expected_status, expected = _one_pass_fixpoint(bounds, factors, c)
            domains = np.array(bounds, dtype=np.int32)
            status = compute_domains_linear_eq_c(
                domains, np.array(factors + [c], dtype=np.int32), np.zeros(1, dtype=np.int32)
            )
            assert (status == PROP_INCONSISTENCY) == (expected_status == PROP_INCONSISTENCY), (bounds, factors, c)
            if status != PROP_INCONSISTENCY:
                assert domains.tolist() == expected, (bounds, factors, c)
                assert (status == PROP_ENTAILMENT) == (expected_status == PROP_ENTAILMENT), (bounds, factors, c)


def _one_pass_fixpoint(bounds: list[tuple[int, int]], factors: list[int], c: int) -> tuple[int, list[list[int]]]:
    """The one-pass rule, with sums computed at the start of each pass, repeated until nothing changes."""
    domains = [list(b) for b in bounds]
    while True:
        sum_min = sum_max = -c
        for (lo, hi), a in zip(domains, factors):
            sum_min += a * hi if a > 0 else a * lo
            sum_max += a * lo if a > 0 else a * hi
        if sum_max > 0 or sum_min < 0:
            return PROP_INCONSISTENCY, domains
        if sum_min == sum_max:
            return PROP_ENTAILMENT, domains
        changed = False
        for d, a in zip(domains, factors):
            if a == 0 or d[0] == d[1]:
                continue
            if a > 0:
                new_min, new_max = d[1] - (sum_min // a), d[0] + (-sum_max // a)
            else:
                new_min, new_max = d[1] - (sum_max // a), d[0] + (-sum_min // a)
            if new_min > d[0]:
                d[0], changed = new_min, True
            if new_max < d[1]:
                d[1], changed = new_max, True
        if not changed:
            return PROP_CONSISTENCY, domains
