from pprint import pprint

from nucs.problems.knapsack_problem import KnapsackProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_FIRST_NON_INSTANTIATED
from nucs.statistics import get_statistics


class TestKnapsack:
    def test_knapsack(self) -> None:
        problem = KnapsackProblem(
            [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
            [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
            55,
        )
        solver = BacktrackSolver(problem, VAR_HEURISTIC_FIRST_NON_INSTANTIATED, DOM_HEURISTIC_MAX_VALUE)
        solution = solver.maximize(problem.weight)
        assert solution
        assert solution[problem.weight] == 54


if __name__ == "__main__":
    problem = KnapsackProblem(
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        55,
    )
    solver = BacktrackSolver(problem, VAR_HEURISTIC_FIRST_NON_INSTANTIATED, DOM_HEURISTIC_MAX_VALUE)
    solution = solver.maximize(problem.weight)
    pprint(get_statistics(problem.statistics))
    print(solution)
