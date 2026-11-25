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
from itertools import product

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_VALUE_PRECEDE
from nucs.propagators.value_precede_propagator import compute_domains_value_precede
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from tests.propagators.propagator_test import PropagatorTest, random_bounds


class TestValuePrecede(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # t (=2) is forbidden up to the first position that can be s (=1); here position 0 cannot be s
            ([(0, 0), (0, 2), (0, 2)], [1, 2], PROP_CONSISTENCY, [[0, 0], [0, 1], [0, 2]]),
            # position 0 is fixed to s -> entailed, nothing pruned afterwards
            ([(1, 1), (0, 2), (2, 2)], [1, 2], PROP_ENTAILMENT, [[1, 1], [0, 2], [2, 2]]),
            # position 0 fixed to t with no possible preceding s -> inconsistency
            ([(2, 2), (0, 2)], [1, 2], PROP_INCONSISTENCY, None),
            # s is fixed at position 1, but position 0 can still be t (=2) inside its domain: not entailed
            ([(1, 4), (0, 0), (0, 4)], [0, 2], PROP_CONSISTENCY, [[1, 4], [0, 0], [0, 4]]),
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
            compute_domains_value_precede, domains, parameters, consistency_result, expected_domains
        )

    @pytest.mark.parametrize("n,d,s,t", [(1, 3, 1, 2), (2, 3, 1, 2), (3, 3, 0, 2), (3, 4, 1, 3), (4, 3, 1, 2)])
    def test_matches_brute_force(self, n: int, d: int, s: int, t: int) -> None:
        def valid(xs) -> bool:  # type: ignore[no-untyped-def]
            return all(s in xs[:i] for i, v in enumerate(xs) if v == t)

        truth = sorted(xs for xs in product(range(d), repeat=n) if valid(xs))
        problem = Problem([(0, d - 1)] * n)
        problem.add_propagator(ALG_VALUE_PRECEDE, range(n), [s, t])
        got = sorted(tuple(solution.tolist()) for solution in BacktrackSolver(problem).find_all())
        assert got == truth

    def test_soundness_against_brute_force(self) -> None:
        # t inside a domain cannot be removed: the entailment must wait until no position before the first s can
        # be t (a fixed s after such a position was declared entailed, and the solver then accepted x = [3, 0])
        rng = random.Random(20261125)
        for _ in range(1000):
            s, t = rng.sample(range(5), 2)

            def is_solution(p: tuple[int, ...], s: int = s, t: int = t) -> bool:
                return all(s in p[:i] for i, v in enumerate(p) if v == t)

            self.assert_sound_against_brute_force(
                compute_domains_value_precede, random_bounds(rng, rng.randint(1, 4), 0, 4), [s, t], is_solution
            )
