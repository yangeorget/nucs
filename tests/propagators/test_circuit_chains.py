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
from nucs.propagators.circuit_chains_propagator import compute_domains_circuit_chains, get_state_circuit_chains
from tests.propagators.propagator_test import PropagatorTest


def _circuits(bounds: list[tuple[int, int]], offset: int) -> list[tuple[int, ...]]:
    """Every successor labelling within the bounds that forms a single circuit over all the nodes."""
    n = len(bounds)
    found = []
    for perm in itertools.permutations(range(n)):
        if any(not lo <= offset + perm[i] <= hi for i, (lo, hi) in enumerate(bounds)):
            continue
        node, length = 0, 0
        while True:
            node = perm[node]
            length += 1
            if node == 0:
                break
        if length == n:
            found.append(tuple(offset + j for j in perm))
    return found


class TestCircuitChains(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # a fixed circuit over three nodes is entailed
            ([(1, 1), (2, 2), (0, 0)], [0], PROP_ENTAILMENT, [[1, 1], [2, 2], [0, 0]]),
            # a closed sub-cycle over two of three nodes
            ([(1, 1), (0, 0), (2, 2)], [0], PROP_INCONSISTENCY, None),
            # 0 -> 1 is fixed: 2 cannot enter 1, and 1 cannot close the chain onto 0 before it spans node 2
            ([(1, 1), (0, 2), (0, 1)], [0], PROP_ENTAILMENT, [[1, 1], [2, 2], [0, 0]]),
            # a single node is its own successor
            ([(0, 0)], [0], PROP_ENTAILMENT, [[0, 0]]),
            # labels offset by 1, self-loops trimmed at the bounds
            ([(1, 3), (1, 3), (1, 3)], [1], PROP_CONSISTENCY, [[2, 3], [1, 3], [1, 2]]),
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
            compute_domains_circuit_chains, domains, parameters, consistency_result, expected_domains
        )

    @pytest.mark.parametrize("offset", [0, 2])
    def test_soundness_against_brute_force(self, offset: int) -> None:
        # never prunes a successor of some circuit, never fails when a circuit exists, and is at its fixpoint after
        # one call
        rng = random.Random(20260917 + offset)
        parameters = np.array([offset], dtype=np.int32)
        for _ in range(3000):
            n = rng.randint(2, 6)
            bounds = []
            for _ in range(n):
                lo = rng.randint(offset - 1, offset + n - 1)
                bounds.append((lo, rng.randint(lo, offset + n)))
            circuits = _circuits(bounds, offset)
            domains = np.array(bounds, dtype=np.int32)
            state = np.zeros(sum(get_state_circuit_chains(n, [offset])), dtype=np.int32)
            status = compute_domains_circuit_chains(domains, parameters, state)
            if status == PROP_INCONSISTENCY:
                assert not circuits, f"declared inconsistent but a circuit exists: {bounds}"
                continue
            for i in range(n):
                for c in circuits:
                    assert domains[i, 0] <= c[i] <= domains[i, 1], f"pruned succ[{i}]={c[i]} of {c}: {bounds}"
            fixpoint = domains.copy()
            status = compute_domains_circuit_chains(domains, parameters, state)
            assert status != PROP_INCONSISTENCY and np.array_equal(domains, fixpoint), f"not idempotent on {bounds}"
