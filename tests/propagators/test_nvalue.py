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
from itertools import product
from typing import List, Optional, Tuple, Union

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.problems.problem import Problem
from nucs.propagators.nvalue_propagator import compute_domains_nvalue
from nucs.propagators.propagators import ALG_NVALUE
from nucs.solvers.backtrack_solver import BacktrackSolver
from tests.propagators.propagator_test import PropagatorTest


class TestNValue(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            # disjoint domains (1,2) and (5,5) force at least 2 distinct values; union {1,2,5} caps at 3
            ([(1, 2), (2, 3), (5, 5), (0, 5)], [], PROP_CONSISTENCY, [[1, 2], [2, 3], [5, 5], [2, 3]]),
            # y forced to 1 makes every x_i equal: intersection of the domains
            ([(0, 4), (2, 6), (1, 5), (1, 1)], [], PROP_CONSISTENCY, [[2, 4], [2, 4], [2, 4], [1, 1]]),
            # two disjoint domains need 2 distinct values but y is capped at 1 -> inconsistency
            ([(0, 1), (3, 4), (1, 1)], [], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        parameters: List[int],
        consistency_result: int,
        expected_domains: Optional[List[List[int]]],
    ) -> None:
        self.assert_compute_domains(compute_domains_nvalue, domains, parameters, consistency_result, expected_domains)

    @pytest.mark.parametrize("n,d", [(1, 3), (2, 3), (3, 3), (3, 4), (4, 3)])
    def test_matches_brute_force(self, n: int, d: int) -> None:
        # for every assignment of the x_i over 0..d-1, the solver must pair it with the true distinct count
        truth = {xs + (len(set(xs)),) for xs in product(range(d), repeat=n)}
        problem = Problem([(0, d - 1)] * n + [(0, n)])
        problem.add_propagator(ALG_NVALUE, range(n + 1))
        got = {tuple(solution.tolist()) for solution in BacktrackSolver(problem).find_all()}
        assert got == truth
