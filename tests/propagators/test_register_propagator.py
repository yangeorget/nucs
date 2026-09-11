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
from collections.abc import Sequence

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MIN_MAX, PROP_CONSISTENCY
from nucs.problems.problem import OFFSETS_STATE, Problem
from nucs.propagators.propagators import ALGORITHM_FLAGS, get_algorithm_nb, register_propagator
from nucs.solvers.backtrack_solver import BacktrackSolver


def get_complexity_leq(n: int, parameters: NDArray) -> int:
    return 1


@njit(cache=True)
def get_triggers_leq(n: int, variable: int, parameters: NDArray) -> int:
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_leq(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """x <= y, as a propagator registered from outside the library."""
    domains[0][DOMAIN_MAX] = min(domains[0][DOMAIN_MAX], domains[1][DOMAIN_MAX])
    domains[1][DOMAIN_MIN] = max(domains[1][DOMAIN_MIN], domains[0][DOMAIN_MIN])
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
        assert len(ALGORITHM_FLAGS) == get_algorithm_nb()  # the flags cover the new algorithm
        problem = Problem([(0, 2), (0, 2)])
        problem.add_propagator(alg, [0, 1])
        solver = BacktrackSolver(problem, log_level="ERROR")  # constructing it is what runs Problem.init
        assert len(problem.algorithm_flags) == get_algorithm_nb()  # the problem's copy covers it too
        solutions = [tuple(solution) for solution in solver.find_all()]
        assert sorted(solutions) == [(x, y) for x in range(3) for y in range(3) if x <= y]


def get_complexity_count_calls(n: int, parameters: NDArray) -> int:
    return 1


@njit(cache=True)
def get_triggers_count_calls(n: int, variable: int, parameters: NDArray) -> int:
    return EVENT_MASK_MIN_MAX


def get_state_count_calls(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    return 1, 0  # one trailed cell, no hint


@njit(cache=True)
def compute_domains_count_calls(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """x <= y, recording in its trailed state cell that it has been entered."""
    prop_state[0] += 1
    domains[0][DOMAIN_MAX] = min(domains[0][DOMAIN_MAX], domains[1][DOMAIN_MAX])
    domains[1][DOMAIN_MIN] = max(domains[1][DOMAIN_MIN], domains[0][DOMAIN_MIN])
    return PROP_CONSISTENCY


class TestPropagatorStateIsBacktracked:
    def test_optim_reset_clears_propagator_state(self) -> None:
        """
        Restarting the search from the root has to put every trailed state block back to its root value.

        OPTIM_RESET drops the trail rather than unwinding it, so nothing restores a block the way a
        backtrack would, and without choice_point_init clearing the trailed prefixes the restarted search
        would begin holding whatever invariant the previous one stopped at. A propagator is entitled to
        read its block as describing the node it is being called at, so that is not a slow search but a
        wrong answer: on knapsack, with the capacity constraint over STATE_MIN_N, it returned 40 instead
        of 54.

        Asserted on the block itself rather than through a search outcome, so that it pins the mechanism
        for every propagator that will ever keep state rather than for one that happens to.
        """
        alg = register_propagator(
            get_triggers_count_calls,
            get_complexity_count_calls,
            compute_domains_count_calls,
            get_state_fct=get_state_count_calls,
        )
        problem = Problem([(0, 2), (0, 2)])
        problem.add_propagator(alg, [0, 1])
        solver = BacktrackSolver(problem, log_level="ERROR")
        assert problem.state_trailed_width == 1
        block = problem.offsets[0, OFFSETS_STATE]
        assert solver.state[block] == 0  # allocated cold
        assert len(list(solver.find_all())) == 6
        assert solver.state[block] > 0  # the search left the block written, and the trail is not unwound
        solver._choice_point_init()  # what OPTIM_RESET does before restarting from the root
        assert solver.state[block] == 0
