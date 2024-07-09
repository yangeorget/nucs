from ncs.heuristics.variable_heuristic import (
    VariableHeuristic,
    min_value_domain_heuristic,
    smallest_domain_variable_heuristic,
)
from ncs.problems.alpha_problem import AlphaProblem
from ncs.solvers.backtrack_solver import BacktrackSolver
from ncs.utils import STATS_SOLVER_SOLUTIONS_NB


class TestAlpha:
    def test_alpha(self) -> None:
        problem = AlphaProblem()
        solver = BacktrackSolver(
            problem, VariableHeuristic(smallest_domain_variable_heuristic, min_value_domain_heuristic)
        )
        solutions = solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 1
        assert solutions[0][:26] == [5, 13, 9, 16, 20, 4, 24, 21, 25, 17, 23, 2, 8, 12, 10, 19,7, 11, 15, 3, 1, 26, 6, 22, 14, 18]


