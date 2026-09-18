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
from nucs.propagators.element_l_eq_c_propagator import compute_domains_element_l_eq_c
from tests.propagators.propagator_test import PropagatorTest, random_bounds


class TestElementLEqC(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(-1, 1), (1, 2), (0, 2)], [1], PROP_CONSISTENCY, [[-1, 1], [1, 2], [0, 1]]),
            ([(-4, -2), (1, 2), (0, 1)], [1], PROP_ENTAILMENT, [[-4, -2], [1, 1], [1, 1]]),
            ([(-4, -2), (1, 2), (0, 1)], [0], PROP_INCONSISTENCY, None),
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
            compute_domains_element_l_eq_c, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        # l holds m values, i ranges one past each end of l
        rng = random.Random(20260918)
        for _ in range(1000):
            m = rng.randint(1, 3)
            c = rng.randint(0, 3)

            def is_solution(p: tuple[int, ...], m: int = m, c: int = c) -> bool:
                return 0 <= p[m] < m and p[p[m]] == c

            self.assert_sound_against_brute_force(
                compute_domains_element_l_eq_c,
                random_bounds(rng, m, 0, 3) + random_bounds(rng, 1, -1, m),
                [c],
                is_solution,
            )
