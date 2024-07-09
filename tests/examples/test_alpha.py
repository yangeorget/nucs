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
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTIONS_NB] == 1
