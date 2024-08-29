from typing import List

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT


class SudokuProblem(Problem):
    """
    A simple model for the sudoku problem.
    """

    def __init__(self, givens: List[List[int]]):
        super().__init__([(1, 9) if given == 0 else (given, given) for line in givens for given in line])
        for i in range(9):
            self.add_propagator((list(range(0 + i * 9, 9 + i * 9)), ALG_ALLDIFFERENT, []))
            self.add_propagator((list(range(0 + i, 81 + i, 9)), ALG_ALLDIFFERENT, []))
        for i in range(3):
            for j in range(3):
                offset = i * 27 + j * 3
                self.add_propagator(
                    (
                        [
                            0 + offset,
                            1 + offset,
                            2 + offset,
                            9 + offset,
                            10 + offset,
                            11 + offset,
                            18 + offset,
                            19 + offset,
                            20 + offset,
                        ],
                        ALG_ALLDIFFERENT,
                        [],
                    )
                )

    def pretty_print_solution(self, solution: List[int]) -> None:
        for i in range(0, 81, 9):
            print(solution[i : i + 9])
