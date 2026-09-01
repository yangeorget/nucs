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

import itertools
import random

import numpy as np
import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_REGULAR
from nucs.propagators.regular_propagator import compute_domains_regular, is_vacuous_regular
from nucs.solvers.backtrack_solver import BacktrackSolver
from tests.propagators.propagator_test import PropagatorTest


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _accepts(seq: tuple[int, ...], q_nb: int, s_nb: int, d: list[int], q0: int, accept: list[int]) -> bool:
    q = q0
    for v in seq:
        q = d[(q - 1) * s_nb + (v - 1)]
        if q == 0:
            return False
    return accept[q - 1] == 1


def _brute_solutions(
    doms: list[tuple[int, int]], q_nb: int, s_nb: int, d: list[int], q0: int, accept: list[int]
) -> list[list[int]]:
    solutions = []
    for seq in itertools.product(*[range(lo, hi + 1) for lo, hi in doms]):
        if _accepts(seq, q_nb, s_nb, d, q0, accept):
            solutions.append(list(seq))
    return solutions


# DFA over {1, 2} accepting words with at least one 2: params [Q, S, q0, d(row-major), accept]
AT_LEAST_ONE_2 = [2, 2, 1, 1, 2, 2, 2, 0, 1]

# Q=1, S=2, q0=1, both symbols self-loop on the single accepting state: accepts every word over {1, 2}
ALL_ACCEPTING = [1, 2, 1, 1, 1, 1]


class TestRegular(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # x0 fixed to 1 so the single remaining position must carry the required 2
            ([(1, 1), (1, 2)], AT_LEAST_ONE_2, PROP_ENTAILMENT, [[1, 1], [2, 2]]),
            # every symbol is 1 -> the automaton never reaches its accepting state
            ([(1, 1), (1, 1)], AT_LEAST_ONE_2, PROP_INCONSISTENCY, None),
            # three free positions: each may still be 1 (another position supplies the 2) -> no pruning
            ([(1, 2), (1, 2), (1, 2)], AT_LEAST_ONE_2, PROP_CONSISTENCY, [[1, 2], [1, 2], [1, 2]]),
            # the automaton accepting every word leaves the domains untouched (Q=1, always in the accept state)
            ([(1, 2), (1, 2)], [1, 2, 1, 1, 1, 1], PROP_CONSISTENCY, [[1, 2], [1, 2]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_regular, domains, parameters, consistency_result, expected_domains)

    def test_binary_is_domain_consistent(self) -> None:
        """Fuzz over random binary-alphabet DFAs: on a binary alphabet the interval domains lose nothing, so the
        propagator is exactly domain-consistent -- each variable must end up as the set of values that appear in
        some accepted word, and an unsatisfiable instance must be detected."""
        rng = random.Random(20260814)
        for _ in range(4000):
            q_nb = rng.randint(1, 4)
            s_nb = 2
            d = [rng.randint(0, q_nb) for _ in range(q_nb * s_nb)]
            q0 = 1
            accept = [rng.randint(0, 1) for _ in range(q_nb)]
            length = rng.randint(1, 4)
            doms = [_pair(rng.randint(1, 2), rng.randint(1, 2)) for _ in range(length)]
            params = [q_nb, s_nb, q0, *d, *accept]
            solutions = _brute_solutions(doms, q_nb, s_nb, d, q0, accept)
            arr = np.array(list(doms), dtype=np.int32)
            status = compute_domains_regular(arr, np.array(params, dtype=np.int32))
            if not solutions:
                assert status == PROP_INCONSISTENCY, (d, accept, doms)
                continue
            assert status != PROP_INCONSISTENCY, (d, accept, doms)
            for var in range(length):
                values = {sol[var] for sol in solutions}
                assert arr[var][DOMAIN_MIN] == min(values) and arr[var][DOMAIN_MAX] == max(values), (
                    d,
                    accept,
                    doms,
                    var,
                )

    @pytest.mark.parametrize(
        "parameters,domains,vacuous",
        [
            (ALL_ACCEPTING, [(1, 2)] * 3, True),  # rejects nothing and every domain is inside the alphabet
            ([2, 2, 1, 1, 2, 2, 1, 1, 1], [(1, 2)] * 3, True),  # two states, total, both accepting
            (ALL_ACCEPTING, [(0, 3)] * 3, False),  # the propagator still trims to the alphabet
            (ALL_ACCEPTING, [(0, 2)] * 3, False),  # a single value below the alphabet is enough
            ([1, 2, 1, 1, 0, 1], [(1, 2)] * 3, False),  # a missing transition rejects that symbol
            ([1, 2, 1, 1, 1, 0], [(1, 2)] * 3, False),  # the only state does not accept
        ],
    )
    def test_regular_vacuity(self, parameters: list[int], domains: list[tuple[int, int]], vacuous: bool) -> None:
        """An automaton rejecting nothing is vacuous only once the domains sit inside its alphabet.

        The propagator keeps each variable within 1..S, so an all-accepting automaton still filters a domain
        that reaches outside it -- which is what stops this from being a parameters-only guard.
        """
        assert is_vacuous_regular(len(domains), parameters, domains) is vacuous
        problem = Problem(list(domains))
        problem.add_propagator(ALG_REGULAR, range(len(domains)), parameters)
        assert problem.propagator_nb == (0 if vacuous else 1)

    @pytest.mark.parametrize(
        "parameters,domains",
        [
            (ALL_ACCEPTING, [(1, 2)] * 3),
            ([2, 2, 1, 1, 2, 2, 1, 1, 1], [(1, 2)] * 3),
        ],
    )
    def test_dropping_a_vacuous_regular_preserves_the_solutions(
        self, parameters: list[int], domains: list[tuple[int, int]]
    ) -> None:
        dropped = Problem(list(domains))
        dropped.add_propagator(ALG_REGULAR, range(len(domains)), parameters)
        assert dropped.propagator_nb == 0
        posted = Problem(list(domains))  # bypass the check so the propagator really runs
        posted.propagators.append((list(range(len(domains))), ALG_REGULAR, list(parameters)))
        posted.propagator_nb += 1
        assert sorted(tuple(s) for s in BacktrackSolver(dropped, log_level="ERROR").find_all()) == sorted(
            tuple(s) for s in BacktrackSolver(posted, log_level="ERROR").find_all()
        )
