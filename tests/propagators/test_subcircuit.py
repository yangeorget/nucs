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
from itertools import permutations

import numpy as np
import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_SUBCIRCUIT
from nucs.propagators.subcircuit_propagator import compute_domains_subcircuit, get_state_subcircuit
from nucs.solvers.backtrack_solver import BacktrackSolver
from tests.propagators.propagator_test import PropagatorTest


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


def _subcircuits(bounds: list[tuple[int, int]], offset: int) -> list[tuple[int, ...]]:
    """Every successor labelling within the bounds that forms a sub-circuit."""
    return [
        tuple(offset + j for j in p)
        for p in permutations(range(len(bounds)))
        if _is_subcircuit(list(p)) and all(lo <= offset + j <= hi for j, (lo, hi) in zip(p, bounds))
    ]


class TestSubcircuit(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # two nodes fixed to the same successor
            ([(1, 1), (1, 1), (0, 2)], [0], PROP_INCONSISTENCY, None),
            # 0 -> 1 is fixed, so 1 cannot be a self-loop and closes the circuit; 2 is left out
            ([(1, 1), (0, 1), (0, 2)], [0], PROP_ENTAILMENT, [[1, 1], [0, 0], [2, 2]]),
            # the same, but 2 cannot be left out: 1 cannot close the chain and has nowhere else to go
            ([(1, 1), (0, 1), (0, 1)], [0], PROP_INCONSISTENCY, None),
            # a self-loop cannot be the successor of another node
            ([(0, 0), (0, 2), (0, 2)], [0], PROP_CONSISTENCY, [[0, 0], [1, 2], [1, 2]]),
            # a fixed cycle makes every other node a self-loop
            ([(1, 1), (0, 0), (2, 3), (2, 3)], [0], PROP_ENTAILMENT, [[1, 1], [0, 0], [2, 2], [3, 3]]),
            # labels offset by 1: 1 cannot be a self-loop, then 2 must close the chain 0 -> 1 -> 2
            ([(2, 2), (2, 3), (1, 3)], [1], PROP_ENTAILMENT, [[2, 2], [3, 3], [1, 1]]),
            # the empty sub-circuit
            ([(0, 0), (1, 1)], [0], PROP_ENTAILMENT, [[0, 0], [1, 1]]),
            # no parameter: 0-based labels
            ([(0, 2), (0, 2), (0, 2)], [], PROP_CONSISTENCY, [[0, 2], [0, 2], [0, 2]]),
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
            compute_domains_subcircuit, domains, parameters, consistency_result, expected_domains
        )

    @pytest.mark.parametrize("offset", [0, 1])
    def test_soundness_against_brute_force(self, offset: int) -> None:
        # never prunes a successor of some sub-circuit, never fails when a sub-circuit exists, and is at its
        # fixpoint after one call, which the one-pass version was not on about 1.5% of these
        rng = random.Random(20260918 + offset)
        parameters = np.array([offset], dtype=np.int32)
        for _ in range(2000):
            n = rng.randint(1, 5)
            bounds = []
            for _ in range(n):
                lo = rng.randint(offset - 1, offset + n - 1)
                bounds.append((lo, rng.randint(lo, offset + n)))
            subcircuits = _subcircuits(bounds, offset)
            domains = np.array(bounds, dtype=np.int32)
            state = np.zeros(sum(get_state_subcircuit(n, [offset])), dtype=np.int32)
            status = compute_domains_subcircuit(domains, parameters, state)
            if status == PROP_INCONSISTENCY:
                assert not subcircuits, f"declared inconsistent but a sub-circuit exists: {bounds}"
                continue
            for i in range(n):
                for c in subcircuits:
                    assert domains[i, 0] <= c[i] <= domains[i, 1], f"pruned succ[{i}]={c[i]} of {c}: {bounds}"
            fixpoint = domains.copy()
            status = compute_domains_subcircuit(domains, parameters, state)
            assert status != PROP_INCONSISTENCY and np.array_equal(domains, fixpoint), f"not idempotent on {bounds}"

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
