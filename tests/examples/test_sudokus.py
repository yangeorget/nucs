from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.heuristics.smallest_domain_variable_heuristic import (
    SmallestDomainVariableHeuristic,
)
from ncs.problems.queens.queens_problem import QueensProblem
from ncs.problems.sudokus.sudoku_problem import SudokuProblem
from ncs.solvers.backtrack_solver import BacktrackSolver


class TestSudokus:
    def test_sudokus_1(self) -> None:
        problem = SudokuProblem([
            [0, 0, 0, 0, 3, 0, 0, 0, 0],
            [2, 8, 9, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 7, 0, 0, 0, 9, 0],
            [0, 0, 0, 0, 0, 0, 8, 0, 6],
            [0, 0, 0, 3, 0, 0, 1, 0, 0],
            [7, 1, 0, 0, 0, 6, 0, 0, 2],
            [0, 6, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 2, 0, 0],
            [0, 0, 1, 0, 5, 0, 6, 0, 0]
        ])

