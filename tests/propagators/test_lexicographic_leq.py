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
# Copyright 2024 - Yan Georget
###############################################################################
from typing import Any

import numpy as np
import pytest

from nucs.constants import PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY, STATS_IDX_SOLVER_SOLUTION_NB
from nucs.numpy_helper import new_parameters_by_values, new_shr_domains_by_values
from nucs.problems.problem import Problem
from nucs.propagators.lexicographic_leq_propagator import compute_domains_lexicographic_leq
from nucs.propagators.propagators import ALG_LEXICOGRAPHIC_LEQ
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestLexicographicLEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(0, 1), 0, 1, 1])
        data = new_parameters_by_values([])
        assert compute_domains_lexicographic_leq(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[0, 1], [0, 0], [1, 1], [1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(0, 1), (0, 1), (0, 1), (0, 1)])
        data = new_parameters_by_values([])
        assert compute_domains_lexicographic_leq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 1], [0, 1], [0, 1], [0, 1]]))

    @pytest.mark.parametrize(
        "values,state",
        [
            ([0, 0, 0, 0], PROP_ENTAILMENT),
            ([0, 0, 0, 1], PROP_ENTAILMENT),
            ([0, 0, 1, 0], PROP_ENTAILMENT),
            ([0, 0, 1, 1], PROP_ENTAILMENT),
            ([0, 1, 0, 0], PROP_INCONSISTENCY),
            ([0, 1, 0, 1], PROP_ENTAILMENT),
            ([0, 1, 1, 0], PROP_ENTAILMENT),
            ([0, 1, 1, 1], PROP_ENTAILMENT),
            ([1, 0, 0, 0], PROP_INCONSISTENCY),
            ([1, 0, 0, 1], PROP_INCONSISTENCY),
            ([1, 0, 1, 0], PROP_ENTAILMENT),
            ([1, 0, 1, 1], PROP_ENTAILMENT),
            ([1, 1, 0, 0], PROP_INCONSISTENCY),
            ([1, 1, 0, 1], PROP_INCONSISTENCY),
            ([1, 1, 1, 0], PROP_INCONSISTENCY),
            ([1, 1, 1, 1], PROP_ENTAILMENT),
        ],
    )
    def test_compute_domains_values(self, values: Any, state: int) -> None:
        domains = new_shr_domains_by_values(values)
        data = new_parameters_by_values([])
        assert compute_domains_lexicographic_leq(domains, data) == state

    def test_solve_1(self) -> None:
        problem = Problem(
            shr_domains_lst=[(0, 1), (0, 1), (0, 1), (0, 1)],
            dom_indices_lst=[0, 1, 2, 3],
            dom_offsets_lst=[0, 0, 0, 0],
        )
        problem.add_propagator(([0, 1, 2, 3], ALG_LEXICOGRAPHIC_LEQ, []))
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 10

    def test_solve_2(self) -> None:
        problem = Problem(
            shr_domains_lst=[(1, 1), (0, 1), (0, 1), (0, 1)],
            dom_indices_lst=[0, 1, 2, 3],
            dom_offsets_lst=[0, 0, 0, 0],
        )
        problem.add_propagator(([0, 1, 2, 3], ALG_LEXICOGRAPHIC_LEQ, []))
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 3
