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
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.cumulative_propagator import compute_domains_cumulative
from tests.propagators.propagator_test import PropagatorTest


def _feasible_starts(
    bounds: List[Tuple[int, int]], durations: List[int], heights: List[int], capacity: int
) -> List[Tuple[int, ...]]:
    """Brute-force every assignment of start times within bounds whose resource profile fits under capacity."""
    ranges = [range(lo, hi + 1) for lo, hi in bounds]
    feasible = []
    for starts in itertools.product(*ranges):
        horizon = max(starts[i] + durations[i] for i in range(len(starts)))
        ok = True
        for t in range(horizon):
            load = sum(heights[i] for i in range(len(starts)) if starts[i] <= t < starts[i] + durations[i])
            if load > capacity:
                ok = False
                break
        if ok:
            feasible.append(starts)
    return feasible


class TestCumulative(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # capacity 2, two unit-height tasks with slack: no compulsory part, no pruning
            ([(0, 10), (0, 10)], [2, 2, 1, 1, 2], PROP_CONSISTENCY, [[0, 10], [0, 10]]),
            # a fixed task occupies [2, 5) at height 1 on a capacity-1 resource: the second task is pushed past it
            ([(2, 2), (1, 10)], [3, 2, 1, 1, 1], PROP_CONSISTENCY, [[2, 2], [5, 10]]),
            # variable heights: a fixed height-2 task fills [0, 4) of a capacity-3 resource, so a height-2 task
            # (2 + 2 > 3) cannot overlap it and is pushed to 4
            ([(0, 0), (0, 10)], [4, 2, 2, 2, 3], PROP_CONSISTENCY, [[0, 0], [4, 10]]),
            # overload: two fixed unit tasks both run during [1, 3) on a capacity-1 resource
            ([(0, 0), (1, 1)], [3, 3, 1, 1, 1], PROP_INCONSISTENCY, None),
            # overload with heights: two fixed height-2 tasks overlap on a capacity-3 resource (2 + 2 > 3)
            ([(0, 0), (0, 0)], [2, 2, 2, 2, 3], PROP_INCONSISTENCY, None),
            # a single task whose demand exceeds the capacity can never run
            ([(0, 5)], [2, 3, 2], PROP_INCONSISTENCY, None),
            # all starts fixed and the profile fits: entailed
            ([(0, 0), (3, 3)], [3, 3, 1, 1, 1], PROP_ENTAILMENT, [[0, 0], [3, 3]]),
            # two height-1 tasks may overlap under capacity 2: fixed and consistent, entailed
            ([(0, 0), (0, 0)], [2, 2, 1, 1, 2], PROP_ENTAILMENT, [[0, 0], [0, 0]]),
            # energetic reasoning, beyond timetabling: neither task has a compulsory part, but over [3, 6) the
            # short task needs 1 of the 3 capacity-1 units, leaving room for only 2 of the long task's 3, so
            # its earliest start is pushed from 3 to 4 (timetabling alone prunes nothing here)
            ([(3, 8), (3, 4)], [3, 1, 1, 1, 1], PROP_CONSISTENCY, [[4, 8], [3, 4]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(
            compute_domains_cumulative, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        # for many small random instances the propagator must be sound: never remove a start that belongs to a
        # feasible schedule, and never claim inconsistency when a feasible schedule exists. (timetabling is
        # incomplete, so it may stay consistent on an infeasible instance -- that is allowed.)
        rng = random.Random(20260622)
        for _ in range(5000):
            n = rng.randint(2, 4)
            capacity = rng.randint(1, 4)
            durations = [rng.randint(1, 3) for _ in range(n)]
            heights = [rng.randint(1, 3) for _ in range(n)]
            bounds = []
            for _ in range(n):
                lo = rng.randint(0, 4)
                hi = lo + rng.randint(0, 4)
                bounds.append((lo, hi))
            feasible = _feasible_starts(bounds, durations, heights, capacity)
            domains = np.array([[lo, hi] for lo, hi in bounds], dtype=np.int32)
            parameters = np.array(durations + heights + [capacity], dtype=np.int32)
            result = compute_domains_cumulative(domains, parameters)
            if result == PROP_INCONSISTENCY:
                assert not feasible, (
                    f"declared inconsistent but feasible: {bounds} p={durations} h={heights} c={capacity}"
                )
                continue
            if not feasible:
                continue  # timetabling is incomplete: staying consistent on an infeasible instance is sound
            for i in range(n):
                bc_min = min(s[i] for s in feasible)
                bc_max = max(s[i] for s in feasible)
                # soundness: the filtered interval must keep every feasible value
                assert domains[i, MIN] <= bc_min, (
                    f"over-pruned MIN of {i}: {bounds} p={durations} h={heights} c={capacity}"
                )
                assert domains[i, MAX] >= bc_max, (
                    f"over-pruned MAX of {i}: {bounds} p={durations} h={heights} c={capacity}"
                )
