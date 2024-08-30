import pytest

from nucs.heuristics.variable_heuristic import (
    VariableHeuristic,
    first_not_instantiated_var_heuristic,
    smallest_domain_var_heuristic,
    split_low_dom_heuristic,
)
from nucs.problems.quasigroup_problem import Quasigroup5Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


class TestQuasigroup:
    @pytest.mark.parametrize(
        "size, solution_nb",
        [
            (7, 3),
            (8, 1),
            (9, 0),
            (10, 0),
            (11, 5),
            # (12, 0),
        ],
    )
    def test_quasigroup5(self, size: int, solution_nb: int) -> None:
        problem = Quasigroup5Problem(size)
        solver = BacktrackSolver(problem, VariableHeuristic(smallest_domain_var_heuristic, split_low_dom_heuristic))
        solver.find_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb
