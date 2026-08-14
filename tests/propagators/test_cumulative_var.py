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

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.propagators.cumulative_propagator import compute_domains_cumulative_var
from tests.propagators.propagator_test import PropagatorTest


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _brute_solutions(
    start_doms: list[tuple[int, int]], dur_doms: list[tuple[int, int]], heights: list[int], capacity: int
) -> list[list[int]]:
    """Enumerates every (start, duration) assignment whose resource profile never exceeds the capacity."""
    n = len(start_doms)
    solutions = []
    start_ranges = [range(lo, hi + 1) for lo, hi in start_doms]
    dur_ranges = [range(lo, hi + 1) for lo, hi in dur_doms]
    for starts in itertools.product(*start_ranges):
        for durations in itertools.product(*dur_ranges):
            horizon_end = max(starts[i] + durations[i] for i in range(n))
            ok = True
            for t in range(min(starts), horizon_end):
                if sum(heights[i] for i in range(n) if starts[i] <= t < starts[i] + durations[i]) > capacity:
                    ok = False
                    break
            if ok:
                solutions.append(list(starts) + list(durations))
    return solutions


class TestCumulativeVar(PropagatorTest):
    # domains = [start_0, ..., start_{n-1}, dur_0, ..., dur_{n-1}]; parameters = [h_0, ..., h_{n-1}, capacity]
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # task 0 fixed at 0 with duration 2 fills [0, 2) at capacity, so task 1 (duration >= 1) starts at 2
            (
                [(0, 0), (0, 5), (2, 2), (1, 1)],
                [1, 1, 1],
                PROP_CONSISTENCY,
                [[0, 0], [2, 5], [2, 2], [1, 1]],
            ),
            # both durations may be 0 (tasks possibly absent): the minimum duration is 0, so nothing is forced
            (
                [(0, 0), (0, 0), (0, 1), (0, 1)],
                [1, 1, 1],
                PROP_CONSISTENCY,
                [[0, 0], [0, 0], [0, 1], [0, 1]],
            ),
            # both tasks fixed at 0 with duration 1 and height 1 overload a capacity of 1
            ([(0, 0), (0, 0), (1, 1), (1, 1)], [1, 1, 1], PROP_INCONSISTENCY, None),
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
            compute_domains_cumulative_var, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        """Fuzz: the propagator must never drop a value that appears in some solution, nor report
        inconsistency when a solution exists (checked against exhaustive enumeration)."""
        rng = random.Random(20260814)
        for _ in range(2000):
            n = rng.randint(1, 3)
            start_doms = [_pair(rng.randint(0, 3), rng.randint(0, 3)) for _ in range(n)]
            dur_doms = [_pair(rng.randint(0, 2), rng.randint(0, 2)) for _ in range(n)]
            heights = [rng.randint(1, 2) for _ in range(n)]
            capacity = rng.randint(1, 3)
            solutions = _brute_solutions(start_doms, dur_doms, heights, capacity)
            arr = np.array(list(start_doms) + list(dur_doms), dtype=np.int32)
            status = compute_domains_cumulative_var(arr, np.array([*heights, capacity], dtype=np.int32))
            if solutions:
                assert status != PROP_INCONSISTENCY, (start_doms, dur_doms, heights, capacity)
                for var in range(2 * n):
                    lo = min(sol[var] for sol in solutions)
                    hi = max(sol[var] for sol in solutions)
                    assert arr[var][MIN] <= lo and arr[var][MAX] >= hi, (start_doms, dur_doms, heights, capacity, var)
