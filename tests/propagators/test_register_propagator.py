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
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY
from nucs.problems.problem import Problem
from nucs.propagators.propagators import IDEMPOTENCIES, get_algorithm_nb, register_propagator
from nucs.solvers.backtrack_solver import BacktrackSolver


def get_complexity_leq(n: int, parameters: NDArray) -> int:
    return 1


@njit(cache=True)
def get_triggers_leq(n: int, variable: int, parameters: NDArray) -> int:
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_leq(domains: NDArray, parameters: NDArray) -> int:
    """x <= y, as a propagator registered from outside the library."""
    domains[0][MAX] = min(domains[0][MAX], domains[1][MAX])
    domains[1][MIN] = max(domains[1][MIN], domains[0][MIN])
    return PROP_CONSISTENCY


class TestRegisterPropagator:
    def test_a_propagator_registered_after_import_can_be_solved_with(self) -> None:
        """A custom propagator gets an algorithm id past the end of every array built at import time.

        The idempotence flags are the array this used to break on: register_propagator rebinds them, because
        np.append returns a new array, so anything that had imported the name by value kept one entry too
        few and indexed past it. Without the JIT that is an IndexError; with it, boundscheck is off and the
        read decides idempotence from whatever follows the array -- and the wrong answer is the unsound one.
        """
        alg = register_propagator(get_triggers_leq, get_complexity_leq, compute_domains_leq)
        assert alg == get_algorithm_nb() - 1
        assert len(IDEMPOTENCIES) == get_algorithm_nb()  # the flags cover the new algorithm
        problem = Problem([(0, 2), (0, 2)])
        problem.add_propagator(alg, [0, 1])
        solver = BacktrackSolver(problem, log_level="ERROR")  # constructing it is what runs Problem.init
        assert len(problem.idempotencies) == get_algorithm_nb()  # the problem's copy covers it too
        solutions = [tuple(solution) for solution in solver.find_all()]
        assert sorted(solutions) == [(x, y) for x in range(3) for y in range(3) if x <= y]
