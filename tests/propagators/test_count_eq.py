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
from nucs.propagators.count_eq_propagator import compute_domains_count_eq
from tests.propagators.propagator_test import PropagatorTest, random_bounds


class TestCountEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 0],
                [5],
                PROP_INCONSISTENCY,
                [],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 1],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5], [1, 1]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 2],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]],
            ),
            (
                [(1, 4), 5, (3, 6), (6, 8), 3, 5, (1, 2)],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [5, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, (-1, 10)],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [1, 3]],
            ),
            (
                [2, (0, 1), (3, 4), 2, 2, (2, 4)],
                [2],
                PROP_ENTAILMENT,
                [[2, 2], [0, 1], [3, 4], [2, 2], [2, 2], [3, 3]],
            ),
            (
                [(3, 5), 5, 5, 5, 3],
                [5],
                PROP_ENTAILMENT,
                [[3, 4], [5, 5], [5, 5], [5, 5], [3, 3]],
            ),
            (
                [(3, 5), (3, 5), 5, (1, 4), 3],
                [5],
                PROP_ENTAILMENT,
                [[5, 5], [5, 5], [5, 5], [1, 4], [3, 3]],
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
        self.assert_compute_domains(compute_domains_count_eq, domains, parameters, consistency_result, expected_domains)

    @pytest.mark.parametrize("seed", range(100))
    @pytest.mark.parametrize("backtrack", [False, True])
    def test_live_set_is_sound(self, seed: int, backtrack: bool) -> None:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 40))  # straddles LIVE_SET_MIN_ARITY, so both paths are exercised
        a = int(rng.integers(0, 5))
        domains = np.empty((n + 1, 2), dtype=np.int32)
        for i in range(n):
            lo = int(rng.integers(0, 5))
            domains[i] = (lo, lo + int(rng.integers(0, 5)))
        domains[n] = (0, n)  # the counter
        self.assert_live_set_is_sound(compute_domains_count_eq, domains, [a], rng, backtrack)

    def test_soundness_against_brute_force(self) -> None:
        # the counter ranges one past each end of what can be counted: a count that cannot reach it used to empty it
        # without failing, and the search went on to return a wrong solution
        rng = random.Random(20260918)
        for _ in range(1000):
            n = rng.randint(1, 4)
            a = rng.randint(0, 2)

            def is_solution(p: tuple[int, ...], a: int = a) -> bool:
                return p[:-1].count(a) == p[-1]

            self.assert_sound_against_brute_force(
                compute_domains_count_eq,
                random_bounds(rng, n, 0, 3) + random_bounds(rng, 1, -1, n + 1),
                [a],
                is_solution,
            )
