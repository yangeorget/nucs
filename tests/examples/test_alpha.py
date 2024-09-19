from pprint import pprint

from nucs.examples.alpha_problem import AlphaProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import STATS_SOLVER_SOLUTION_NB, get_statistics


class TestAlpha:
    def test_alpha(self) -> None:
        problem = AlphaProblem()
        solver = BacktrackSolver(
            problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
        )
        solutions = solver.find_all()
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == 1
        assert solutions[0][:26] == [
            5,
            13,
            9,
            16,
            20,
            4,
            24,
            21,
            25,
            17,
            23,
            2,
            8,
            12,
            10,
            19,
            7,
            11,
            15,
            3,
            1,
            26,
            6,
            22,
            14,
            18,
        ]


if __name__ == "__main__":
    problem = AlphaProblem()
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    solutions = solver.find_all()
    pprint(get_statistics(solver.statistics))
    print(solutions[0])
