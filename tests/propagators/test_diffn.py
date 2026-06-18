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
from nucs.propagators.diffn_propagator import compute_domains_diffn
from tests.propagators.propagator_test import PropagatorTest


def _feasible_placements(bounds: List[Tuple[int, int]], dx: List[int], dy: List[int]) -> List[Tuple[int, ...]]:
    """Brute-force every placement of the rectangles (within bounds) with no pairwise overlap."""
    n = len(dx)
    ranges = [range(lo, hi + 1) for lo, hi in bounds]
    feasible = []
    for coords in itertools.product(*ranges):
        x = coords[:n]
        y = coords[n:]
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if not (x[i] + dx[i] <= x[j] or x[j] + dx[j] <= x[i] or y[i] + dy[i] <= y[j] or y[j] + dy[j] <= y[i]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            feasible.append(coords)
    return feasible


class TestDiffn(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # two 2x2 squares with plenty of room: no pruning (x0,x1,y0,y1)
            ([(0, 9), (0, 9), (0, 9), (0, 9)], [2, 2, 2, 2], PROP_CONSISTENCY, [[0, 9], [0, 9], [0, 9], [0, 9]]),
            # rectangle 0 fixed at origin (2x2); rectangle 1 (2x2) forced to overlap in y around [0,1] -> must
            # be to the right: x1 >= 2
            ([(0, 0), (0, 5), (0, 0), (0, 1)], [2, 2, 2, 2], PROP_CONSISTENCY, [[0, 0], [2, 5], [0, 0], [0, 1]]),
            # both squares pinned to the same cell -> overlap -> inconsistency
            ([(0, 0), (0, 0), (0, 0), (0, 0)], [2, 2, 2, 2], PROP_INCONSISTENCY, None),
            # fixed, non-overlapping placement -> entailed
            ([(0, 0), (2, 2), (0, 0), (0, 0)], [2, 2, 2, 2], PROP_ENTAILMENT, [[0, 0], [2, 2], [0, 0], [0, 0]]),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_diffn, domains, parameters, consistency_result, expected_domains)

    def test_soundness_against_brute_force(self) -> None:
        # for many small random instances the propagator must be sound: never remove a coordinate value that
        # belongs to a feasible non-overlapping placement, and never claim inconsistency when one exists.
        rng = random.Random(20260618)
        for _ in range(4000):
            n = rng.randint(2, 3)
            dx = [rng.randint(1, 3) for _ in range(n)]
            dy = [rng.randint(1, 3) for _ in range(n)]
            bounds = []
            for _ in range(2 * n):  # x bounds then y bounds
                lo = rng.randint(0, 3)
                hi = lo + rng.randint(0, 3)
                bounds.append((lo, hi))
            feasible = _feasible_placements(bounds, dx, dy)
            domains = np.array([[lo, hi] for lo, hi in bounds], dtype=np.int32)
            result = compute_domains_diffn(domains, np.array(dx + dy, dtype=np.int32))
            if result == PROP_INCONSISTENCY:
                assert not feasible, f"declared inconsistent but feasible: {bounds} {dx} {dy}"
                continue
            if not feasible:
                continue  # the pairwise filter is incomplete: staying consistent here is still sound
            for v in range(2 * n):
                bc_min = min(coord[v] for coord in feasible)
                bc_max = max(coord[v] for coord in feasible)
                assert domains[v, MIN] <= bc_min, f"over-pruned MIN of {v}: {bounds} {dx} {dy}"
                assert domains[v, MAX] >= bc_max, f"over-pruned MAX of {v}: {bounds} {dx} {dy}"
