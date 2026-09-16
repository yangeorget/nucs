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

import numpy as np
import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.propagators.mul_eq_propagator import compute_domains_mul_eq
from tests.propagators.propagator_test import PropagatorTest


class TestMulEq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # z tightened to the product hull; x, y already bound-consistent
            ([(2, 3), (4, 5), (0, 100)], [], PROP_CONSISTENCY, [[2, 3], [4, 5], [8, 15]]),
            # x and y both forced from z (2 * 3 = 6) -> entailment
            ([(0, 10), (3, 3), (6, 6)], [], PROP_ENTAILMENT, [[2, 2], [3, 3], [6, 6]]),
            # negative factor: z hull is negative, x and y stay tight
            ([(-3, -2), (4, 5), (-100, 100)], [], PROP_CONSISTENCY, [[-3, -2], [4, 5], [-15, -8]]),
            # 2 * 3 != 7 -> inconsistency
            ([(2, 2), (3, 3), (7, 7)], [], PROP_INCONSISTENCY, None),
            # y straddles 0: x is not tightened, but z and y are
            ([(2, 4), (-1, 2), (-100, 100)], [], PROP_CONSISTENCY, [[2, 4], [-1, 2], [-4, 8]]),
            # x fixed to 0 forces z to 0 and entails the constraint for any y
            ([(0, 0), (5, 9), (-100, 100)], [], PROP_ENTAILMENT, [[0, 0], [5, 9], [0, 0]]),
            # division rounds inward: z in [7, 11], y = 3 -> x in [3, 3] (ceil(7/3)=3, floor(11/3)=3),
            # then z is re-tightened to 3 * 3 = 9 and the constraint is entailed
            ([(0, 10), (3, 3), (7, 11)], [], PROP_ENTAILMENT, [[3, 3], [3, 3], [9, 9]]),
            # x = 1 / -2 has no integer value: the empty quotient fails instead of dividing by a crossed x
            ([(-1, 0), (-2, -2), (1, 1)], [], PROP_INCONSISTENCY, None),
            ([(-2, -2), (-1, 0), (1, 1)], [], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_mul_eq, domains, parameters, consistency_result, expected_domains)

    def test_soundness_against_brute_force(self) -> None:
        # over every small domain triple spanning 0, iterated to its fixpoint as the engine iterates a non-idempotent
        # propagator: it never raises, never fails when a solution exists, and never prunes a value of one
        for x0, x1, y0, y1, z0, z1 in itertools.product(range(-3, 4), repeat=6):
            if x0 > x1 or y0 > y1 or z0 > z1:
                continue
            bounds = [(x0, x1), (y0, y1), (z0, z1)]
            solutions = [(x, y, x * y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1) if z0 <= x * y <= z1]
            domains = np.array(bounds, dtype=np.int32)
            for _ in range(50):
                before = domains.copy()
                status = compute_domains_mul_eq(domains, np.empty(0, dtype=np.int32), np.zeros(1, dtype=np.int32))
                if status != PROP_CONSISTENCY or np.array_equal(before, domains):
                    break
            if status == PROP_INCONSISTENCY:
                assert not solutions, f"declared inconsistent but {solutions[0]} satisfies {bounds}"
                continue
            for solution in solutions:
                for v, value in enumerate(solution):
                    assert domains[v, 0] <= value <= domains[v, 1], f"pruned {solution} from {bounds}"
