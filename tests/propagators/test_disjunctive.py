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
from nucs.propagators.disjunctive_propagator import compute_domains_disjunctive
from tests.propagators.propagator_test import PropagatorTest, random_bounds


class TestDisjunctive(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # plenty of room: no pruning
            ([(0, 10), (0, 10)], [2, 2], PROP_CONSISTENCY, [[0, 10], [0, 10]]),
            # edge finding: task 0 has a compulsory part forcing task 1 to start no earlier than 3
            ([(0, 2), (0, 5)], [3, 3], PROP_CONSISTENCY, [[0, 2], [3, 5]]),
            # overload: three length-2 tasks cannot all fit before time 5
            ([(0, 3), (0, 3), (0, 3)], [2, 2, 2], PROP_INCONSISTENCY, None),
            # all start times fixed and non-overlapping: entailed
            ([(0, 0), (2, 2)], [2, 2], PROP_ENTAILMENT, [[0, 0], [2, 2]]),
            # not-first/not-last: task 0 is forced after the fixed task 1 (start 2), then task 2 cannot be
            # anything but last, so its start is raised to 4 -- a prune edge finding needs a second pass for
            ([(0, 2), (1, 1), (2, 4)], [2, 1, 1], PROP_ENTAILMENT, [[2, 2], [1, 1], [4, 4]]),
            # detectable precedence 0 << 2 (task 2 cannot precede task 0), so task 0 must complete before
            # task 2's latest start 2, lowering task 0's latest start to 1 -- a prune neither edge finding nor
            # not-first/not-last makes (even at fixpoint)
            ([(0, 2), (6, 6), (1, 2)], [1, 3, 3], PROP_CONSISTENCY, [[0, 1], [6, 6], [1, 2]]),
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
            compute_domains_disjunctive, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        # edge finding is incomplete, so staying consistent on an infeasible instance is allowed
        rng = random.Random(20260618)
        for _ in range(3000):
            n = rng.randint(2, 4)
            durations = [rng.randint(1, 3) for _ in range(n)]

            def is_solution(starts: tuple[int, ...], n: int = n, durations: list[int] = durations) -> bool:
                return all(
                    starts[i] + durations[i] <= starts[j] or starts[j] + durations[j] <= starts[i]
                    for i in range(n)
                    for j in range(i + 1, n)
                )

            self.assert_sound_against_brute_force(
                compute_domains_disjunctive, random_bounds(rng, n, 0, 7), durations, is_solution
            )
