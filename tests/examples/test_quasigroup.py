import argparse
from pprint import pprint

import pytest

from nucs.heuristics.variable_heuristic import VariableHeuristic, min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.problems.quasigroup_problem import Quasigroup5Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_SOLVER_SOLUTION_NB, get_statistics


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
        solver = BacktrackSolver(problem, VariableHeuristic(smallest_domain_var_heuristic, min_value_dom_heuristic))
        solver.find_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=9)
    args = parser.parse_args()
    problem = Quasigroup5Problem(args.n)
    solver = BacktrackSolver(problem, VariableHeuristic(smallest_domain_var_heuristic, min_value_dom_heuristic))
    solver.find_all()
    pprint(get_statistics(problem.statistics))
