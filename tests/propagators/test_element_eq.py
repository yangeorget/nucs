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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.element_eq_propagator import compute_domains_element_eq
from tests.propagators.propagator_test import PropagatorTest, random_bounds


class TestElementEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(0, 4), (-2, 2)], [3, 0, 1, 2, 4], PROP_CONSISTENCY, [[1, 3], [0, 2]]),
            ([(0, 4), (-2, -1)], [3, 0, 1, 2, 4], PROP_INCONSISTENCY, []),
            ([(0, 4), (-2, 0)], [3, 0, 1, 2, 4], PROP_ENTAILMENT, [[1, 1], [0, 0]]),
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
            compute_domains_element_eq, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        # i ranges one past each end of l: an i left with no index used to empty v's bounds without failing, and the
        # search then looped forever
        rng = random.Random(20260918)
        for _ in range(1000):
            m = rng.randint(1, 4)
            l = [rng.randint(0, 4) for _ in range(m)]

            def is_solution(p: tuple[int, ...], m: int = m, l: list[int] = l) -> bool:
                return 0 <= p[0] < m and l[p[0]] == p[1]

            self.assert_sound_against_brute_force(
                compute_domains_element_eq, random_bounds(rng, 1, -1, m) + random_bounds(rng, 1, -1, 5), l, is_solution
            )
