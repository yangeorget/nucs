from nucs.examples.sudoku_problem import SudokuProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import STATS_SOLVER_SOLUTION_NB


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
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == 1

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
        assert solver.statistics[STATS_SOLVER_SOLUTION_NB] == 1
