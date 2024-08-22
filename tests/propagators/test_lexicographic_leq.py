from typing import Any

import numpy as np
import pytest

from nucs.memory import (
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    new_data_by_values,
    new_domains_by_values,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_LEXICOGRAPHIC_LEQ, compute_domains
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


class TestLexicographicLEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_domains_by_values([(0, 1), 0, 1, 1])
        data = new_data_by_values([])
        assert compute_domains(ALG_LEXICOGRAPHIC_LEQ, domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[0, 1], [0, 0], [1, 1], [1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_domains_by_values([(0, 1), (0, 1), (0, 1), (0, 1)])
        data = new_data_by_values([])
        assert compute_domains(ALG_LEXICOGRAPHIC_LEQ, domains, data) == PROP_CONSISTENCY
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
        domains = new_domains_by_values(values)
        data = new_data_by_values([])
        assert compute_domains(ALG_LEXICOGRAPHIC_LEQ, domains, data) == state

    def test_solve_1(self) -> None:
        problem = Problem(
            shr_domains=[(0, 1), (0, 1), (0, 1), (0, 1)], dom_indices=[0, 1, 2, 3], dom_offsets=[0, 0, 0, 0]
        )
        problem.set_propagators([([0, 1, 2, 3], ALG_LEXICOGRAPHIC_LEQ, [])])
        solver = BacktrackSolver(problem)
        solver.find_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == 10

    def test_solve_2(self) -> None:
        problem = Problem(
            shr_domains=[(1, 1), (0, 1), (0, 1), (0, 1)], dom_indices=[0, 1, 2, 3], dom_offsets=[0, 0, 0, 0]
        )
        problem.set_propagators([([0, 1, 2, 3], ALG_LEXICOGRAPHIC_LEQ, [])])
        solver = BacktrackSolver(problem)
        for solution in solver.solve():
            print(solution)
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == 3
