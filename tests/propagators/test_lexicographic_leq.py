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
# Copyright 2024-2025 - Yan Georget
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
        problem = Problem(
            domains=[(0, 1), (0, 1), (0, 1), (0, 1)],
            variables=[0, 1, 2, 3],
            offsets=[0, 0, 0, 0],
        )
        problem.add_propagator(([0, 1, 2, 3], ALG_LEXICOGRAPHIC_LEQ, []))
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 10

    def test_solve_2(self) -> None:
        problem = Problem(
            domains=[(1, 1), (0, 1), (0, 1), (0, 1)],
            variables=[0, 1, 2, 3],
            offsets=[0, 0, 0, 0],
        )
        problem.add_propagator(([0, 1, 2, 3], ALG_LEXICOGRAPHIC_LEQ, []))
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLUTION_NB] == 3
