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

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
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

    def test_soundness_against_brute_force(self) -> None:
        # over many small random instances the propagator must never claim inconsistency when a feasible
        # ground assignment exists, and never prune a value that belongs to a feasible assignment.
        rng = random.Random(20260809)
        for _ in range(5000):
            b = rng.randint(1, 4)
            bounds = []
            for _ in range(2 * b + 1):  # b conditions, b values, then y -- each 0/1 or undecided
                lo, hi = rng.choice([(0, 0), (1, 1), (0, 1)])
                bounds.append((lo, hi))
            feasible = _feasible(bounds)
            domains = np.array([[lo, hi] for lo, hi in bounds], dtype=np.int32)
            # if_then_else is not idempotent: iterate as the engine does before judging the outcome
            parameters = np.empty(0, dtype=np.int32)
            result = compute_domains_if_then_else(domains, parameters)
            while result == PROP_CONSISTENCY:
                previous = domains.copy()
                result = compute_domains_if_then_else(domains, parameters)
                if np.array_equal(previous, domains):
                    break
            if result == PROP_INCONSISTENCY:
                assert not feasible, f"declared inconsistent but feasible exists: {bounds}"
                continue
            assert feasible, f"stayed consistent but no feasible assignment: {bounds}"
            for v in range(2 * b + 1):
                bc_min = min(a[v] for a in feasible)
                bc_max = max(a[v] for a in feasible)
                assert domains[v, MIN] <= bc_min, f"over-pruned MIN of var {v}: {bounds}"
                assert domains[v, MAX] >= bc_max, f"over-pruned MAX of var {v}: {bounds}"
