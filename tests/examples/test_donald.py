from pprint import pprint

from nucs.problems.donald_problem import DonaldProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.statistics import STATS_SOLVER_SOLUTION_NB, get_statistics


class TestDonald:
    def test_donald(self) -> None:
        problem = DonaldProblem()
        solver = BacktrackSolver(problem, VAR_HEURISTIC_SMALLEST_DOMAIN, DOM_HEURISTIC_MIN_VALUE)
        solutions = solver.solve_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == 1
        assert solutions[0] == [4, 3, 5, 9, 1, 8, 6, 2, 7, 0]


if __name__ == "__main__":
    problem = DonaldProblem()
    solver = BacktrackSolver(problem, VAR_HEURISTIC_SMALLEST_DOMAIN, DOM_HEURISTIC_MIN_VALUE)
    for solution in solver.solve():
        problem.pretty_print_solution(solution)
    pprint(get_statistics(problem.statistics))
