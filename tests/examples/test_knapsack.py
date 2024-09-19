from pprint import pprint

from nucs.examples.knapsack_problem import KnapsackProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import first_not_instantiated_var_heuristic, max_value_dom_heuristic
from nucs.statistics import get_statistics


class TestKnapsack:
    def test_knapsack(self) -> None:
        problem = KnapsackProblem(
            [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
            [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
            55,
        )
        solver = BacktrackSolver(
            problem, var_heuristic=first_not_instantiated_var_heuristic, dom_heuristic=max_value_dom_heuristic
        )
        solution = solver.maximize(problem.weight)
        assert solution
        assert solution[problem.weight] == 54


if __name__ == "__main__":
    problem = KnapsackProblem(
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        [40, 40, 38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28, 26, 26, 24, 24, 22, 22],
        55,
    )
    solver = BacktrackSolver(
        problem, var_heuristic=first_not_instantiated_var_heuristic, dom_heuristic=max_value_dom_heuristic
    )
    solution = solver.maximize(problem.weight)
    pprint(get_statistics(solver.statistics))
    print(solution)
