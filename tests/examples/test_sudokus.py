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
# Copyright 2024-2026 - Yan Georget
###############################################################################
import json

import pytest

from nucs.examples.sudoku.sudoku_problem import SudokuProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_IDX_SOLUTION_NB


class TestSudokus:
    @pytest.mark.parametrize("path", ["datasets/examples/sudoku/sudoku1.json", "datasets/examples/sudoku/sudoku2.json"])
    def test_solve_all(self, path: str) -> None:
        with open(path, "r") as json_file:
            givens = json.load(json_file)["givens"]
            problem = SudokuProblem(givens)
            solver = BacktrackSolver(problem)
            solver.solve_all()
            assert solver.statistics[STATS_IDX_SOLUTION_NB] == 1
