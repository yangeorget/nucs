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

from nucs.constants import STATS_IDX_SOLUTION_NB
from nucs.examples.sudoku.sudoku_problem import SudokuProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestSudokus:
    @pytest.mark.parametrize("path", ["datasets/sudoku/sudoku1.json", "datasets/sudoku/sudoku1.json"])
    def test_sudokus(self, path: str) -> None:
        with open(path, "r") as json_file:
            givens = json.load(json_file)["givens"]
            problem = SudokuProblem(givens)
            solver = BacktrackSolver(problem)
            solver.solve_all()
            assert solver.statistics[STATS_IDX_SOLUTION_NB] == 1
