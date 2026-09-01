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

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.propagators.bin_packing_load_propagator import compute_domains_bin_packing_load
from tests.propagators.propagator_test import PropagatorTest


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _brute_solutions(
    weights: list[int], load_doms: list[tuple[int, int]], bin_doms: list[tuple[int, int]]
) -> list[list[int]]:
    """Enumerates every assignment of the bins (offset 1) whose resulting loads fit the load domains."""
    bin_nb = len(load_doms)
    solutions = []
    for assignment in itertools.product(*[range(lo, hi + 1) for lo, hi in bin_doms]):
        loads = [0] * bin_nb
        for i, b in enumerate(assignment):
            loads[b - 1] += weights[i]
        if all(load_doms[j][0] <= loads[j] <= load_doms[j][1] for j in range(bin_nb)):
            solutions.append(loads + list(assignment))
    return solutions


class TestBinPackingLoad(PropagatorTest):
    # domains = [load_bin1, load_bin2, ...loads..., bin_item0, bin_item1, ...]; parameters = [bin_offset=1, w...]
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # all items placed -> each load is fully determined
            (
                [(0, 100), (0, 100), (1, 1), (2, 2), (1, 1)],
                [1, 3, 2, 5],
                PROP_CONSISTENCY,
                [[8, 8], [2, 2], [1, 1], [2, 2], [1, 1]],
            ),
            # one free item -> loads get required (fixed items) lower and possible (candidates) upper bounds
            (
                [(0, 100), (0, 100), (1, 2), (1, 1), (2, 2)],
                [1, 3, 2, 5],
                PROP_CONSISTENCY,
                [[2, 5], [5, 8], [1, 2], [1, 1], [2, 2]],
            ),
            # bin 1 capacity 4 cannot hold either weight-5 item -> both pushed to bin 2 (overflow pruning)
            (
                [(0, 4), (0, 100), (1, 2), (1, 2)],
                [1, 5, 5],
                PROP_CONSISTENCY,
                [[0, 0], [10, 10], [2, 2], [2, 2]],
            ),
            # bin 1 needs load >= 4 and only item 0 (w=5) can supply it -> item 0 forced into bin 1
            (
                [(4, 100), (0, 100), (1, 2), (1, 2)],
                [1, 5, 3],
                PROP_CONSISTENCY,
                [[5, 8], [0, 3], [1, 1], [1, 2]],
            ),
            # no-sum load tightening: both weight-5 items may go to bin 1, so its load can only be 0, 5 or 10;
            # a domain of [1, 9] therefore collapses to exactly 5
            (
                [(1, 9), (0, 100), (1, 2), (1, 2)],
                [1, 5, 5],
                PROP_CONSISTENCY,
                [[5, 5], [5, 5], [1, 2], [1, 2]],
            ),
            # no-sum infeasibility: bin 1 load fixed to 4 but subsets of {5, 5} only reach 0, 5, 10
            ([(4, 4), (0, 100), (1, 2), (1, 2)], [1, 5, 5], PROP_INCONSISTENCY, None),
            # item fixed to bin 1 forces load 5 but that load is capped at 3 -> inconsistency
            ([(0, 3), (0, 100), (1, 1)], [1, 5], PROP_INCONSISTENCY, None),
            # already tight and consistent -> no change
            (
                [(8, 8), (2, 2), (1, 1), (2, 2), (1, 1)],
                [1, 3, 2, 5],
                PROP_CONSISTENCY,
                [[8, 8], [2, 2], [1, 1], [2, 2], [1, 1]],
            ),
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
            compute_domains_bin_packing_load, domains, parameters, consistency_result, expected_domains
        )

    def test_soundness_against_brute_force(self) -> None:
        """Fuzz: the propagator must never drop a value that appears in some solution, nor report
        inconsistency when a solution exists (checked against exhaustive enumeration)."""
        rng = random.Random(20260814)
        for _ in range(4000):
            bin_nb = rng.randint(1, 3)
            item_nb = rng.randint(1, 5)
            weights = [rng.randint(1, 4) for _ in range(item_nb)]
            total = sum(weights)
            load_doms = [_pair(rng.randint(0, total), rng.randint(0, total)) for _ in range(bin_nb)]
            bin_doms = [_pair(rng.randint(1, bin_nb), rng.randint(1, bin_nb)) for _ in range(item_nb)]
            solutions = _brute_solutions(weights, load_doms, bin_doms)
            arr = np.array(list(load_doms) + list(bin_doms), dtype=np.int32)
            status = compute_domains_bin_packing_load(arr, np.array([1, *weights], dtype=np.int32))
            # only soundness is asserted: with no solution the propagator may or may not detect it (the exact
            # subset-sum reasoning is complete within its budget but the budget can be exceeded)
            if solutions:
                assert status != PROP_INCONSISTENCY, (weights, load_doms, bin_doms)
                for var in range(bin_nb + item_nb):
                    lo = min(sol[var] for sol in solutions)
                    hi = max(sol[var] for sol in solutions)
                    assert arr[var][DOMAIN_MIN] <= lo and arr[var][DOMAIN_MAX] >= hi, (
                        weights,
                        load_doms,
                        bin_doms,
                        var,
                    )
