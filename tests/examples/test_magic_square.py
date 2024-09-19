import pytest

from nucs.examples.magic_square_problem import MagicSquareProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import max_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


class TestMagicSquare:

    def test_first_diagonal(self) -> None:
        assert MagicSquareProblem(3).first_diag() == [0, 4, 8]

    def test_second_diagonal(self) -> None:
        assert MagicSquareProblem(3).second_diag() == [6, 4, 2]

    @pytest.mark.parametrize("size,solution_nb", [(2, 0), (3, 1), (4, 880)])
    def test_magic_square(self, size: int, solution_nb: int) -> None:
        problem = MagicSquareProblem(size)
        solver = BacktrackSolver(
            problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=max_value_dom_heuristic
        )
        solver.solve_all()
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == solution_nb
