import pytest

from nucs.problems.magic_square_problem import MagicSquareProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MAX_VALUE, VAR_HEURISTIC_SMALLEST_DOMAIN
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


class TestMagicSquare:

    def test_first_diagonal(self) -> None:
        assert MagicSquareProblem(3).first_diag() == [0, 4, 8]

    def test_second_diagonal(self) -> None:
        assert MagicSquareProblem(3).second_diag() == [6, 4, 2]

    @pytest.mark.parametrize("size,solution_nb", [(2, 0), (3, 1), (4, 880)])
    def test_magic_square(self, size: int, solution_nb: int) -> None:
        problem = MagicSquareProblem(size)
        solver = BacktrackSolver(problem, VAR_HEURISTIC_SMALLEST_DOMAIN, DOM_HEURISTIC_MAX_VALUE)
        solver.find_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb
