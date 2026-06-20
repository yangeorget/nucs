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
from nucs.propagators.disjunctive_propagator import compute_domains_disjunctive
from tests.propagators.propagator_test import PropagatorTest


def _feasible_starts(bounds: List[Tuple[int, int]], durations: List[int]) -> List[Tuple[int, ...]]:
    """Brute-force every assignment of start times within bounds where no two tasks overlap."""
    ranges = [range(lo, hi + 1) for lo, hi in bounds]
    feasible = []
    for starts in itertools.product(*ranges):
        ok = True
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                if not (starts[i] + durations[i] <= starts[j] or starts[j] + durations[j] <= starts[i]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            feasible.append(starts)
    return feasible


class TestDisjunctive(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # plenty of room: no pruning
            ([(0, 10), (0, 10)], [2, 2], PROP_CONSISTENCY, [[0, 10], [0, 10]]),
            # edge finding: task 0 has a compulsory part forcing task 1 to start no earlier than 3
            ([(0, 2), (0, 5)], [3, 3], PROP_CONSISTENCY, [[0, 2], [3, 5]]),
            # overload: three length-2 tasks cannot all fit before time 5
            ([(0, 3), (0, 3), (0, 3)], [2, 2, 2], PROP_INCONSISTENCY, None),
            # all start times fixed and non-overlapping: entailed
            ([(0, 0), (2, 2)], [2, 2], PROP_ENTAILMENT, [[0, 0], [2, 2]]),
            # not-first/not-last: task 0 is forced after the fixed task 1 (start 2), then task 2 cannot be
            # anything but last, so its start is raised to 4 -- a prune edge finding needs a second pass for
            ([(0, 2), (1, 1), (2, 4)], [2, 1, 1], PROP_CONSISTENCY, [[2, 2], [1, 1], [4, 4]]),
            # detectable precedence 0 << 2 (task 2 cannot precede task 0), so task 0 must complete before
            # task 2's latest start 2, lowering task 0's latest start to 1 -- a prune neither edge finding nor
            # not-first/not-last makes (even at fixpoint)
            ([(0, 2), (6, 6), (1, 2)], [1, 3, 3], PROP_CONSISTENCY, [[0, 1], [6, 6], [1, 2]]),
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
            compute_domains_disjunctive, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        # for many small random instances the propagator must be sound: never remove a value that belongs to
        # a feasible non-overlapping schedule, and never claim inconsistency when a feasible schedule exists.
        # (edge finding is incomplete, so it may stay consistent on an infeasible instance -- that is allowed.)
        rng = random.Random(20260618)
        for _ in range(5000):
            n = rng.randint(2, 4)
            durations = [rng.randint(1, 3) for _ in range(n)]
            bounds = []
            for _ in range(n):
                lo = rng.randint(0, 5)
                hi = lo + rng.randint(0, 5)
                bounds.append((lo, hi))
            feasible = _feasible_starts(bounds, durations)
            domains = np.array([[lo, hi] for lo, hi in bounds], dtype=np.int32)
            result = compute_domains_disjunctive(domains, np.array(durations, dtype=np.int32))
            if result == PROP_INCONSISTENCY:
                assert not feasible, f"declared inconsistent but feasible: {bounds} {durations} {feasible[:3]}"
                continue
            if not feasible:
                continue  # edge finding is incomplete: staying consistent on an infeasible instance is sound
            for i in range(n):
                bc_min = min(s[i] for s in feasible)
                bc_max = max(s[i] for s in feasible)
                # soundness: the filtered interval must keep every feasible value
                assert domains[i, MIN] <= bc_min, f"over-pruned MIN of {i}: {bounds} {durations}"
                assert domains[i, MAX] >= bc_max, f"over-pruned MAX of {i}: {bounds} {durations}"
