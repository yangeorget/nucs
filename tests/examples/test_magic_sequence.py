import pytest

from nucs.problems.magic_sequence_problem import MagicSequenceProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import DOM_HEURISTIC_MIN_VALUE, VAR_HEURISTIC_LAST_NON_INSTANTIATED
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


class TestMagicSequence:

    @pytest.mark.parametrize("size,zero_nb", [(50, 46), (100, 96), (200, 196)])
    def test_magic_sequence(self, size: int, zero_nb: int) -> None:
        problem = MagicSequenceProblem(size)
        solver = BacktrackSolver(problem, VAR_HEURISTIC_LAST_NON_INSTANTIATED, DOM_HEURISTIC_MIN_VALUE)
        solutions = solver.solve_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == 1
        assert solutions[0][0] == zero_nb
