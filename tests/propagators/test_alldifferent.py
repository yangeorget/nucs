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

import numpy as np
import pytest

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY
from nucs.propagators.alldifferent_propagator import (
    SORT_MAX_N,
    argsort_into,
    argsort_into_warm,
    compute_domains_alldifferent,
    path_max,
    path_min,
    path_set,
)
from tests.propagators.propagator_test import PropagatorTest


class TestAlldifferent(PropagatorTest):
    def test_path_min(self) -> None:
        a = np.array([2, 3, 4, 0, 1])
        assert path_min(a, 0) == 0
        assert path_min(a, 1) == 1
        assert path_min(a, 2) == 2
        assert path_min(a, 3) == 0
        assert path_min(a, 4) == 1

    def test_path_max(self) -> None:
        a = np.array([2, 3, 4, 0, 1])
        assert path_max(a, 0) == 4
        assert path_max(a, 1) == 3
        assert path_max(a, 2) == 4
        assert path_max(a, 3) == 3
        assert path_max(a, 4) == 4

    def test_path_set(self) -> None:
        a = np.array([2, 3, 4, 0, 1])
        path_set(a, 0, 4, -1)
        assert np.all(a == np.array([-1, 3, -1, 0, 1]))

    # both sides of SORT_MAX_N, since each argsort takes the np.argsort fallback above it, and a stale seed
    # (a permutation of a *previous* call's keys) is the case the warm path exists for and the one that
    # could leave it unsorted
    @pytest.mark.parametrize("n", [1, 2, 5, SORT_MAX_N, SORT_MAX_N + 1, 3 * SORT_MAX_N])
    @pytest.mark.parametrize("bound", [DOMAIN_MIN, DOMAIN_MAX])
    def test_argsort_into_warm(self, n: int, bound: int) -> None:
        rng = np.random.default_rng(n * 2 + bound)
        # a seed that is a permutation of range(n) but decorrelated from the keys, as after a backtrack
        seed = rng.permutation(n).astype(np.int32)
        for _ in range(4):
            mins = rng.integers(0, 3 * n + 1, size=n, dtype=np.int32)
            domains = np.stack([mins, mins + rng.integers(0, n + 1, size=n, dtype=np.int32)], axis=1)
            domains = np.ascontiguousarray(domains, dtype=np.int32)
            cold = np.empty(n, dtype=np.int32)
            argsort_into(cold, domains, bound)
            argsort_into_warm(seed, domains, bound)
            # the two orderings agree on the keys -- not necessarily on the permutation, since ties break by
            # whatever order each started from -- and the warm result is still a permutation, which is what
            # the next call warm-starts from
            assert np.array_equal(domains[seed, bound], domains[cold, bound])
            assert np.array_equal(np.sort(seed), np.arange(n))

    # above SORT_MAX_N the warm sort runs on a shift budget: a descent stays inside it and the insertion
    # sort finishes, a jump blows it and hands over to np.argsort. Both branches have to land on the same
    # ordering, and test_argsort_into_warm only ever reaches the second -- its seed is always decorrelated.
    @pytest.mark.parametrize("moved", [0, 1, 8, -1])  # -1 decorrelates the seed, which blows the budget
    @pytest.mark.parametrize("bound", [DOMAIN_MIN, DOMAIN_MAX])
    def test_argsort_into_warm_above_sort_max_n(self, moved: int, bound: int) -> None:
        n = 4 * SORT_MAX_N
        rng = np.random.default_rng(100 + moved * 2 + bound)
        keys = rng.integers(0, 4 * n, size=n, dtype=np.int32)
        seed = np.argsort(keys).astype(np.int32)  # sorted for the keys as they stood at the previous call
        if moved < 0:
            rng.shuffle(seed)
        else:
            for _ in range(moved):  # a few bounds move, as one node of a descent does
                keys[rng.integers(0, n)] += rng.integers(1, 3 * n)
        other = keys + rng.integers(0, n + 1, size=n, dtype=np.int32)
        columns = (keys, other) if bound == DOMAIN_MIN else (other - 2 * n, keys)
        domains = np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.int32)
        cold = np.empty(n, dtype=np.int32)
        argsort_into(cold, domains, bound)
        argsort_into_warm(seed, domains, bound)
        assert np.array_equal(domains[seed, bound], domains[cold, bound])
        assert np.array_equal(np.sort(seed), np.arange(n))

    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            (
                [(3, 6), (3, 4), (2, 5), (2, 4), (3, 4), (1, 6)],
                [],
                PROP_CONSISTENCY,
                [[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]],
            ),
            ([(0, 0), (2, 2), (1, 2)], [], PROP_CONSISTENCY, [[0, 0], [2, 2], [1, 1]]),
            ([(0, 0), (0, 4), (0, 4), (0, 4), (0, 4)], [], PROP_CONSISTENCY, [[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]]),
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
            compute_domains_alldifferent, domains, parameters, consistency_result, expected_domains
        )
