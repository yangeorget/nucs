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

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.count_eq_propagator import compute_domains_count_eq, get_state_count_eq
from tests.propagators.propagator_test import PropagatorTest


class TestCountEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 0],
                [5],
                PROP_INCONSISTENCY,
                [],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 1],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5], [1, 1]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 2],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]],
            ),
            (
                [(1, 4), 5, (3, 6), (6, 8), 3, 5, (1, 2)],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [5, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]],
            ),
            (
                [(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, (-1, 10)],
                [5],
                PROP_CONSISTENCY,
                [[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [1, 3]],
            ),
            (
                [2, (0, 1), (3, 4), 2, 2, (2, 4)],
                [2],
                PROP_ENTAILMENT,
                [[2, 2], [0, 1], [3, 4], [2, 2], [2, 2], [3, 3]],
            ),
            (
                [(3, 5), 5, 5, 5, 3],
                [5],
                PROP_ENTAILMENT,
                [[3, 4], [5, 5], [5, 5], [5, 5], [3, 3]],
            ),
            (
                [(3, 5), (3, 5), 5, (1, 4), 3],
                [5],
                PROP_ENTAILMENT,
                [[5, 5], [5, 5], [5, 5], [1, 4], [3, 3]],
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
        self.assert_compute_domains(compute_domains_count_eq, domains, parameters, consistency_result, expected_domains)

    @pytest.mark.parametrize("seed", range(40))
    def test_live_set_agrees_with_a_cold_call(self, seed: int) -> None:
        """
        A warm call has to reach the same conclusion as a cold one on the same domains.

        This propagator carries its counts and the set of x_i still undetermined across calls, rebuilding
        neither, which is sound only because domains narrow monotonically down a branch. So the thing to
        check is not any one call but a *chain* of them: narrow the domains a little at a time, letting one
        state block follow the chain, and demand at each step that it agrees -- status and every bound --
        with a propagator handed the same domains and a block it has never seen. A live set that fell out of
        step with the domains would show up here and nowhere in the single-call cases above.
        """
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 12))
        a = int(rng.integers(0, 5))
        parameters = np.array([a], dtype=np.int32)
        domains = np.empty((n + 1, 2), dtype=np.int32)
        for i in range(n):
            lo = int(rng.integers(0, 5))
            domains[i] = (lo, lo + int(rng.integers(0, 5)))
        domains[n] = (0, n)
        trailed_nb, hint_nb = get_state_count_eq(n + 1, [a])
        warm_state = np.zeros(trailed_nb + hint_nb, dtype=np.int32)
        warm_domains = domains.copy()
        for _ in range(12):
            # narrow one bound of one variable -- only narrow, because a widening is something the
            # engine never does inside a branch and the whole live set is unsound without that
            i = int(rng.integers(0, n + 1))
            if warm_domains[i, 0] < warm_domains[i, 1]:
                if rng.integers(0, 2):
                    warm_domains[i, 0] += 1
                else:
                    warm_domains[i, 1] -= 1
            cold_domains = warm_domains.copy()
            cold_state = np.zeros(trailed_nb + hint_nb, dtype=np.int32)
            warm_state[trailed_nb] = cold_state[trailed_nb] = 1  # what the engine pre-sets
            warm_status = compute_domains_count_eq(warm_domains, parameters, warm_state)
            cold_status = compute_domains_count_eq(cold_domains, parameters, cold_state)
            assert warm_status == cold_status
            assert np.array_equal(warm_domains, cold_domains)
            assert warm_state[trailed_nb] == cold_state[trailed_nb]  # and the same change report
            if warm_status != PROP_CONSISTENCY:
                break

    @pytest.mark.parametrize("seed", range(40))
    def test_live_set_survives_a_backtrack(self, seed: int) -> None:
        """
        Restoring what the engine restores has to restore the live set.

        Only the two trailed cells are undone on backtrack; the permutation behind them is a hint and is
        left exactly as the abandoned subtree permuted it. That is the sparse set's whole bargain, and it
        holds because an x_i leaving the live prefix is *parked* at its far end rather than dropped, so the
        places between the restored size and the current one hold precisely the ones that left. Here a chain
        descends, the trailed cells and the domains are rolled back the way the trail rolls them back, and a
        different chain descends from the same point -- with a cold propagator shadowing every call.
        """
        rng = np.random.default_rng(1000 + seed)
        n = int(rng.integers(3, 12))
        a = int(rng.integers(0, 5))
        parameters = np.array([a], dtype=np.int32)
        domains = np.empty((n + 1, 2), dtype=np.int32)
        for i in range(n):
            lo = int(rng.integers(0, 5))
            domains[i] = (lo, lo + int(rng.integers(0, 5)))
        domains[n] = (0, n)
        trailed_nb, hint_nb = get_state_count_eq(n + 1, [a])
        warm_state = np.zeros(trailed_nb + hint_nb, dtype=np.int32)

        def narrow(doms: np.ndarray) -> None:
            i = int(rng.integers(0, n + 1))
            if doms[i, 0] < doms[i, 1]:
                if rng.integers(0, 2):
                    doms[i, 0] += 1
                else:
                    doms[i, 1] -= 1

        def step(doms: np.ndarray) -> int:
            cold_domains = doms.copy()
            cold_state = np.zeros(trailed_nb + hint_nb, dtype=np.int32)
            warm_state[trailed_nb] = cold_state[trailed_nb] = 1
            warm_status = compute_domains_count_eq(doms, parameters, warm_state)
            cold_status = compute_domains_count_eq(cold_domains, parameters, cold_state)
            assert warm_status == cold_status
            assert np.array_equal(doms, cold_domains)
            assert warm_state[trailed_nb] == cold_state[trailed_nb]
            return warm_status

        node_domains = domains.copy()
        for _ in range(3):
            narrow(node_domains)
            if step(node_domains) != PROP_CONSISTENCY:
                return
        # the choice point: the trail holds the trailed cells and the bounds, and nothing else
        trailed_mark = warm_state[:trailed_nb].copy()
        domains_mark = node_domains.copy()
        for _ in range(6):  # descend, permuting the live array as it goes
            narrow(node_domains)
            if step(node_domains) != PROP_CONSISTENCY:
                break
        warm_state[:trailed_nb] = trailed_mark  # backtrack: exactly what the trail undoes
        node_domains = domains_mark.copy()
        for _ in range(6):  # and away down a different branch
            narrow(node_domains)
            if step(node_domains) != PROP_CONSISTENCY:
                return
