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
from nucs.problems.circuit_problem import CircuitProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestCircuitProblem:
    @pytest.mark.parametrize("size, solution_nb", [(2, 1), (3, 2), (4, 6)])
    def test_circuit(self, size: int, solution_nb: int) -> None:
        problem = CircuitProblem(size)
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == solution_nb