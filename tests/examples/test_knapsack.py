from nucs.examples.knapsack.knapsack_problem import KnapsackProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import first_not_instantiated_var_heuristic, max_value_dom_heuristic


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
