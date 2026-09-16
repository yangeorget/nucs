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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.if_then_else_propagator import compute_domains_if_then_else
from tests.propagators.propagator_test import PropagatorTest


def _feasible(bounds: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    """Brute-force every ground assignment of (conditions, values, y) satisfying the if-then-else."""
    b = (len(bounds) - 1) // 2
    ranges = [range(lo, hi + 1) for lo, hi in bounds]
    feasible = []
    for a in itertools.product(*ranges):
        c = a[:b]
        x = a[b : 2 * b]
        y = a[2 * b]
        k = -1
        for i in range(b):
            if c[i] == 1:
                k = i
                break
        if k == -1 or y == x[k]:  # no branch taken -> y free; else y must equal the taken branch's value
            feasible.append(a)
    return feasible


class TestIfThenElse(PropagatorTest):
    @pytest.mark.parametrize(
        # layout: [c0, c1, x0, x1, y]
        "domains,parameters,consistency_result,expected_domains",
        [
            # branch 0 taken (c0 fixed true): y == x0 == 1, both ground -> entailed
            ([(1, 1), (0, 1), (1, 1), (0, 1), (0, 1)], [], PROP_ENTAILMENT, [[1, 1], [0, 1], [1, 1], [0, 1], [1, 1]]),
            # every condition false -> no branch taken -> y unconstrained -> entailed, no pruning
            ([(0, 0), (0, 0), (0, 1), (0, 1), (0, 1)], [], PROP_ENTAILMENT, [[0, 0], [0, 0], [0, 1], [0, 1], [0, 1]]),
            # branch 0 taken but y=1 while x0=0 -> inconsistency
            ([(1, 1), (0, 1), (0, 0), (0, 1), (1, 1)], [], PROP_INCONSISTENCY, None),
            # c0 undecided, y=1 & x0=0 disagree -> c0 forced 0, then branch 1 (true) taken -> y == x1 == 1
            ([(0, 1), (1, 1), (0, 0), (1, 1), (1, 1)], [], PROP_ENTAILMENT, [[0, 0], [1, 1], [0, 0], [1, 1], [1, 1]]),
            # c1 guaranteed true and both candidate values are 1 -> y = 1 whichever branch is taken
            ([(0, 1), (1, 1), (1, 1), (1, 1), (0, 1)], [], PROP_CONSISTENCY, [[0, 1], [1, 1], [1, 1], [1, 1], [1, 1]]),
            # nothing decided -> no pruning
            ([(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], [], PROP_CONSISTENCY, [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
            # y = if c then x else 0: y excludes 0, so the else branch cannot be taken -> c holds and x = y
            ([(0, 1), (1, 1), (3, 4), (0, 0), (3, 3)], [], PROP_ENTAILMENT, [[1, 1], [1, 1], [3, 3], [0, 0], [3, 3]]),
            # neither branch can equal y although one must be taken -> inconsistency
            ([(0, 1), (1, 1), (4, 6), (0, 0), (1, 1)], [], PROP_INCONSISTENCY, None),
            # both branches yield 0 -> y = 0 whichever is taken
            ([(0, 1), (1, 1), (0, 0), (0, 0), (0, 1)], [], PROP_CONSISTENCY, [[0, 1], [1, 1], [0, 0], [0, 0], [0, 0]]),
            # y lies within the hull of the candidate values
            ([(0, 1), (1, 1), (2, 5), (8, 9), (0, 10)], [], PROP_CONSISTENCY, [[0, 1], [1, 1], [2, 5], [8, 9], [2, 9]]),
            # a branch that cannot be taken is no candidate, whatever its condition may be
            ([(0, 1), (0, 1), (5, 5), (1, 2), (0, 3)], [], PROP_CONSISTENCY, [[0, 0], [0, 1], [5, 5], [1, 2], [0, 3]]),
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
            compute_domains_if_then_else, domains, parameters, consistency_result, expected_domains
        )

    @pytest.mark.parametrize("values_max", [1, 3])
    def test_bound_consistency_against_brute_force(self, values_max: int) -> None:
        # on distinct variables, iterated to its fixpoint as the engine iterates a non-idempotent propagator, it
        # leaves exactly the bounds of the feasible ground assignments: it fails exactly when there are none, never
        # prunes a supported value, and leaves no unsupported bound
        rng = random.Random(20260916 + values_max)
        parameters = np.empty(0, dtype=np.int32)
        for _ in range(1500):
            b = rng.randint(1, 3)
            bounds = [rng.choice([(0, 0), (1, 1), (0, 1)]) for _ in range(b)]  # the conditions
            for _ in range(b + 1):  # the values, then y
                lo = rng.randint(0, values_max)
                bounds.append((lo, rng.randint(lo, values_max)))
            feasible = _feasible(bounds)
            domains = np.array([[lo, hi] for lo, hi in bounds], dtype=np.int32)
            for _ in range(4 * (2 * b + 1) * (values_max + 1) + 2):  # each changing pass narrows some bound
                before = domains.copy()
                result = compute_domains_if_then_else(domains, parameters, np.empty(0, dtype=np.int32))
                if result != PROP_CONSISTENCY or np.array_equal(before, domains):
                    break
            else:
                raise AssertionError(f"no fixpoint on {bounds}")
            if result == PROP_INCONSISTENCY:
                assert not feasible, f"declared inconsistent but feasible exists: {bounds}"
                continue
            assert feasible, f"stayed consistent but no feasible assignment: {bounds}"
            for v in range(2 * b + 1):
                expected = [min(a[v] for a in feasible), max(a[v] for a in feasible)]
                assert domains[v].tolist() == expected, f"var {v} of {bounds}: {domains[v].tolist()} != {expected}"
