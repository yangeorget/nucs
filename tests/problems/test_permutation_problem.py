###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024 - Yan Georget
###############################################################################
import pytest

from nucs.constants import STATS_IDX_SOLVER_SOLUTION_NB
from nucs.problems.permutation_problem import PermutationProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestPermutationProblem:
    @pytest.mark.parametrize("size, solution_nb", [(2, 2), (3, 6), (4, 24)])
    def test_solve_all(self, size: int, solution_nb: int) -> None:
        problem = PermutationProblem(size)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb

    def test_solve(self) -> None:
        problem = PermutationProblem(5)
        solver = BacktrackSolver(problem)
        it = solver.solve()
        solution = next(it)
        assert solution.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        solution = next(it)
        assert solution.tolist() == [0, 1, 2, 4, 3, 0, 1, 2, 4, 3]
