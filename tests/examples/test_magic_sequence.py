import pytest

from nucs.heuristics.variable_heuristic import (
    VariableHeuristic,
    last_not_instantiated_var_heuristic,
    min_value_dom_heuristic,
)
from nucs.problems.magic_sequence_problem import MagicSequenceProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


class TestMagicSequence:

    @pytest.mark.parametrize("size,zero_nb", [(50, 46), (100, 96), (200, 196)])
    def test_magic_sequence(self, size: int, zero_nb: int) -> None:
        problem = MagicSequenceProblem(size)
        solver = BacktrackSolver(
            problem, VariableHeuristic(last_not_instantiated_var_heuristic, min_value_dom_heuristic)
        )
        solutions = solver.solve_all()
        assert problem.statistics[STATS_SOLVER_SOLUTION_NB] == 1
        assert solutions[0][0] == zero_nb
