from pprint import pprint

from nucs.heuristics.variable_heuristic import (
    VariableHeuristic,
    min_value_dom_heuristic,
    smallest_domain_var_heuristic,
)
from nucs.problems.donald_problem import DonaldProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_SOLVER_SOLUTION_NB, get_statistics


class TestDonald:
    def test_donald(self) -> None:
        problem = DonaldProblem()
        solver = BacktrackSolver(problem, VariableHeuristic(smallest_domain_var_heuristic, min_value_dom_heuristic))
        solutions = solver.solve_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == 1
        assert solutions[0] == [4, 3, 5, 9, 1, 8, 6, 2, 7, 0]


if __name__ == "__main__":
    problem = DonaldProblem()
    solver = BacktrackSolver(problem, VariableHeuristic(smallest_domain_var_heuristic, min_value_dom_heuristic))
    for solution in solver.solve():
        problem.pretty_print_solution(solution)
    pprint(get_statistics(problem.statistics))
