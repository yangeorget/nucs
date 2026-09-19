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
from nucs.propagators.propagators import ALG_VALUE_PRECEDE_CHAIN
from nucs.propagators.value_precede_chain_propagator import (
    compute_domains_value_precede_chain,
    value_precede_chain_parameters,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from tests.propagators.propagator_test import PropagatorTest, random_bounds


def is_chained(xs: tuple[int, ...], chain: list[int]) -> bool:
    """Returns whether every chain value that occurs in xs first occurs after the one before it in the chain."""
    firsts = [xs.index(c) if c in xs else len(xs) for c in chain]
    return all(firsts[m - 1] < firsts[m] or firsts[m] == len(xs) for m in range(1, len(chain)))


class TestValuePrecedeChain(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,chain,consistency_result,expected_domains",
        [
            # only 1 can start the chain, then at most one step up per position
            ([(1, 3), (1, 3), (1, 3)], [1, 2, 3], PROP_CONSISTENCY, [[1, 1], [1, 2], [1, 3]]),
            # 3 at position 0 is followed by 4, outside the chain, so the lower bound rises to it
            ([(2, 4), (0, 5)], [1, 2, 3], PROP_CONSISTENCY, [[4, 4], [0, 5]]),
            # a chain value with no possible predecessor, fixed -> inconsistency
            ([(0, 0), (2, 2)], [1, 2], PROP_INCONSISTENCY, None),
            # a fixed prefix realising the whole chain -> entailed
            ([(1, 1), (0, 0), (2, 2), (0, 5)], [1, 2], PROP_ENTAILMENT, [[1, 1], [0, 0], [2, 2], [0, 5]]),
            # an unordered chain: 3 first, then 1, then 2
            ([(1, 3), (1, 3), (1, 3)], [3, 1, 2], PROP_CONSISTENCY, [[3, 3], [1, 3], [1, 3]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        chain: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        parameters = value_precede_chain_parameters(chain)
        self.assert_compute_domains(
            compute_domains_value_precede_chain, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        rng = random.Random(20260919)
        for _ in range(600):
            n = rng.randint(1, 4)
            chain = rng.sample(range(5), rng.randint(2, 4))  # unordered, with values of the box left out

            def is_solution(p: tuple[int, ...], chain: list[int] = chain) -> bool:
                return is_chained(p, chain)

            self.assert_sound_against_brute_force(
                compute_domains_value_precede_chain,
                random_bounds(rng, n, 0, 4),
                value_precede_chain_parameters(chain),
                is_solution,
            )

    @pytest.mark.parametrize("n,d,chain", [(3, 4, [1, 2, 3]), (4, 3, [0, 1, 2]), (4, 4, [2, 0, 3])])
    def test_find_all(self, n: int, d: int, chain: list[int]) -> None:
        truth = sorted(xs for xs in product(range(d), repeat=n) if is_chained(xs, chain))
        problem = Problem([(0, d - 1)] * n)
        problem.add_propagator(ALG_VALUE_PRECEDE_CHAIN, range(n), value_precede_chain_parameters(chain))
        got = sorted(tuple(solution.tolist()) for solution in BacktrackSolver(problem).find_all())
        assert got == truth
