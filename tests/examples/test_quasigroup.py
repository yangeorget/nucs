import pytest

from nucs.examples.quasigroup.quasigroup_problem import Quasigroup5Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


class TestQuasigroup:
    @pytest.mark.parametrize(
        "size, solution_nb",
        [
            (7, 3),
            (8, 1),
            (9, 0),
            (10, 0),
            # (11, 5),
            # (12, 0),
        ],
    )
    def test_quasigroup5(self, size: int, solution_nb: int) -> None:
        problem = Quasigroup5Problem(size)
        solver = BacktrackSolver(
            problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
        )
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb
