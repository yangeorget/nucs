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
from nucs.constants import STATS_IDX_SOLVER_SOLUTION_NB
from nucs.examples.sudoku.sudoku_problem import SudokuProblem
from nucs.solvers.backtrack_solver import BacktrackSolver


class TestSudokus:
    def test_sudokus_1(self) -> None:
        problem = SudokuProblem(
            [
                [0, 0, 0, 0, 3, 0, 0, 0, 0],
                [2, 8, 9, 0, 0, 0, 0, 0, 0],
                [0, 0, 5, 7, 0, 0, 0, 9, 0],
                [0, 0, 0, 0, 0, 0, 8, 0, 6],
                [0, 0, 0, 3, 0, 0, 1, 0, 0],
                [7, 1, 0, 0, 0, 6, 0, 0, 2],
                [0, 6, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 2, 0, 0],
                [0, 0, 1, 0, 5, 0, 6, 0, 0],
            ]
        )
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 1

    def test_sudokus_2(self) -> None:
        problem = SudokuProblem(
            [
                [6, 0, 0, 0, 1, 0, 0, 8, 0],
                [5, 1, 7, 4, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 5, 0, 0, 3, 0, 0],
                [1, 6, 0, 0, 0, 9, 0, 5, 2],
                [2, 5, 9, 6, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 7, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 0, 4, 0, 0],
            ]
        )
        solver = BacktrackSolver(problem)
        solver.solve_all()
        assert solver.statistics[STATS_IDX_SOLVER_SOLUTION_NB] == 1
