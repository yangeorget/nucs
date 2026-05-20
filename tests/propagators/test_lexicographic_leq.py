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
from typing import List, Optional, Tuple, Union

import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY, STATS_IDX_SOLUTION_NB
from nucs.problems.problem import Problem
from nucs.propagators.lexicographic_leq_propagator import compute_domains_lexicographic_leq
from nucs.propagators.propagators import ALG_LEXICOGRAPHIC_LEQ
from nucs.solvers.backtrack_solver import BacktrackSolver
from tests.propagators.propagator_test import PropagatorTest


class TestLexicographicLeq(PropagatorTest):
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(0, 1), 0, 1, 1], [], PROP_ENTAILMENT, [[0, 1], [0, 0], [1, 1], [1, 1]]),
            ([(0, 1), (0, 1), (0, 1), (0, 1)], [], PROP_CONSISTENCY, [[0, 1], [0, 1], [0, 1], [0, 1]]),
            ([0, 0, 0, 0], [], PROP_ENTAILMENT, None),
            ([0, 0, 0, 1], [], PROP_ENTAILMENT, None),
            ([0, 0, 1, 0], [], PROP_ENTAILMENT, None),
            ([0, 0, 1, 1], [], PROP_ENTAILMENT, None),
            ([0, 1, 0, 0], [], PROP_INCONSISTENCY, None),
            ([0, 1, 0, 1], [], PROP_ENTAILMENT, None),
            ([0, 1, 1, 0], [], PROP_ENTAILMENT, None),
            ([0, 1, 1, 1], [], PROP_ENTAILMENT, None),
            ([1, 0, 0, 0], [], PROP_INCONSISTENCY, None),
            ([1, 0, 0, 1], [], PROP_INCONSISTENCY, None),
            ([1, 0, 1, 0], [], PROP_ENTAILMENT, None),
            ([1, 0, 1, 1], [], PROP_ENTAILMENT, None),
            ([1, 1, 0, 0], [], PROP_INCONSISTENCY, None),
            ([1, 1, 0, 1], [], PROP_INCONSISTENCY, None),
            ([1, 1, 1, 0], [], PROP_INCONSISTENCY, None),
            ([1, 1, 1, 1], [], PROP_ENTAILMENT, None),
            # n=3, exercises compute_domains_2 while loop body (all 4 equal at position 1)
            (
                [(0, 2), 5, (0, 2), (0, 2), 5, (0, 2)],
                [],
                PROP_CONSISTENCY,
                [[0, 2], [5, 5], [0, 2], [0, 2], [5, 5], [0, 2]],
            ),
            # n=3, routes to compute_domains_2's x[i,MIN] > y[i,MAX] branch (forces xq < yq)
            (
                [(0, 2), (3, 5), 0, (0, 2), (0, 2), 5],
                [],
                PROP_CONSISTENCY,
                [[0, 1], [3, 5], [0, 0], [1, 2], [0, 2], [5, 5]],
            ),
            # n=3, routes to compute_domains_3 via x[i,MAX] == y[i,MIN] branch
            (
                [(0, 2), (0, 2), 0, (0, 2), (2, 5), 5],
                [],
                PROP_CONSISTENCY,
                [[0, 2], [0, 2], [0, 0], [0, 2], [2, 5], [5, 5]],
            ),
            # n=4, exercises compute_domains_3 while loop body
            (
                [(0, 2), (0, 2), 5, 0, (0, 2), (2, 5), 5, 5],
                [],
                PROP_CONSISTENCY,
                [[0, 2], [0, 2], [5, 5], [0, 0], [0, 2], [2, 5], [5, 5], [5, 5]],
            ),
            # n=4, exercises compute_domains_4 while loop body and forces xq < yq
            (
                [(0, 2), (2, 5), 5, 5, (0, 2), (0, 2), 5, 0],
                [],
                PROP_CONSISTENCY,
                [[0, 1], [2, 5], [5, 5], [5, 5], [1, 2], [0, 2], [5, 5], [0, 0]],
            ),
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
            compute_domains_lexicographic_leq, domains, parameters, consistency_result, expected_domains
        )

    def test_solve_1(self) -> None:
        problem = Problem(domains=[(0, 1), (0, 1), (0, 1), (0, 1)])
        problem.add_propagator(ALG_LEXICOGRAPHIC_LEQ, [0, 1, 2, 3])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 10

    def test_solve_2(self) -> None:
        problem = Problem(domains=[(1, 1), (0, 1), (0, 1), (0, 1)])
        problem.add_propagator(ALG_LEXICOGRAPHIC_LEQ, [0, 1, 2, 3])
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 3
