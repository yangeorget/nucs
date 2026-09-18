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
from nucs.propagators.element_l_eq_c_alldifferent_propagator import compute_domains_element_l_eq_c_alldifferent
from tests.propagators.propagator_test import PropagatorTest, random_bounds


class TestElementLEqCAlldifferent(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # exactly one l[idx] is the singleton c -> i is set, entailment
            ([5, 3, 7, (0, 2)], [3], PROP_ENTAILMENT, [[5, 5], [3, 3], [7, 7], [1, 1]]),
            # all l[idx] disjoint from c -> i becomes empty, inconsistency
            ([5, 6, 7, (0, 2)], [3], PROP_INCONSISTENCY, None),
            # disjoint l[idx] at the end -> i[DOMAIN_MAX] tightened then l[i] = c, entailment
            ([(1, 5), (10, 15), (0, 1), (0, 2)], [3], PROP_ENTAILMENT, [[3, 3], [10, 15], [0, 1], [0, 0]]),
            # disjoint l[idx] at the start -> i[DOMAIN_MIN] increased then l[i] = c, entailment
            ([(10, 15), (1, 5), (0, 1), (0, 2)], [3], PROP_ENTAILMENT, [[10, 15], [3, 3], [0, 1], [1, 1]]),
            # multiple l[idx] intersect c, no entailment
            ([(1, 5), (2, 6), (0, 1)], [3], PROP_CONSISTENCY, [[1, 5], [2, 6], [0, 1]]),
            # i given out of bounds gets clamped
            ([(1, 5), (2, 6), (-5, 10)], [3], PROP_CONSISTENCY, [[1, 5], [2, 6], [0, 1]]),
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
            compute_domains_element_l_eq_c_alldifferent, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        # l holds m values, i ranges one past each end of l; l is all different, as the alldifferent it relies on
        # makes it
        rng = random.Random(20260918)
        for _ in range(2000):
            m = rng.randint(1, 3)
            c = rng.randint(0, 3)

            def is_solution(p: tuple[int, ...], m: int = m, c: int = c) -> bool:
                return 0 <= p[m] < m and p[p[m]] == c

            def is_all_different(p: tuple[int, ...], m: int = m) -> bool:
                return len(set(p[:m])) == m

            self.assert_sound_against_brute_force(
                compute_domains_element_l_eq_c_alldifferent,
                random_bounds(rng, m, 0, 3) + random_bounds(rng, 1, -1, m),
                [c],
                is_solution,
                is_all_different,
            )
