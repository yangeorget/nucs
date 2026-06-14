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
from itertools import permutations

import pytest

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_SUBCIRCUIT
from nucs.solvers.backtrack_solver import BacktrackSolver


def _is_subcircuit(p) -> bool:  # type: ignore[no-untyped-def]
    """A 0-based successor permutation is a sub-circuit iff the nodes with p[i] != i form a single cycle."""
    active = [i for i in range(len(p)) if p[i] != i]
    if not active:
        return True
    seen = set()
    i = active[0]
    while i not in seen:
        seen.add(i)
        i = p[i]
    return seen == set(active) and i == active[0]


class TestSubcircuit:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_matches_brute_force(self, n: int) -> None:
        # alldifferent + subcircuit must enumerate exactly the true sub-circuits, with no duplicates
        truth = sorted(p for p in permutations(range(n)) if _is_subcircuit(list(p)))
        problem = Problem([(0, n - 1)] * n)
        problem.add_propagator(ALG_ALLDIFFERENT, range(n))
        problem.add_propagator(ALG_SUBCIRCUIT, range(n))
        solutions = sorted(tuple(solution.tolist()) for solution in BacktrackSolver(problem).find_all())
        assert solutions == truth

    def test_two_disjoint_cycles_rejected(self) -> None:
        # [1, 0, 3, 2] is two 2-cycles -> not a single sub-circuit
        problem = Problem([(1, 1), (0, 0), (3, 3), (2, 2)])
        problem.add_propagator(ALG_ALLDIFFERENT, range(4))
        problem.add_propagator(ALG_SUBCIRCUIT, range(4))
        assert next(BacktrackSolver(problem).solve(), None) is None
