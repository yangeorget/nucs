from typing import List, Tuple

from ncs.problems.problem import ALG_ALLDIFFERENT, Problem


class SudokuProblem(Problem):
    """
    A simple model for the sudoku problem.
    """

    def __init__(self, givens: List[List[int]]):
        super().__init__(
            shared_domains=[(1, 9) if given == 0 else (given, given) for line in givens for given in line],
            domain_indices=list(range(81)),
            domain_offsets=[0] * 81,
        )
        propagators: List[Tuple[List[int], int, List[int]]] = []
        for i in range(9):
            propagators.append((list(range(0 + i * 9, 9 + i * 9)), ALG_ALLDIFFERENT, []))
            propagators.append((list(range(0 + i, 81 + i, 9)), ALG_ALLDIFFERENT, []))
        for i in range(3):
            for j in range(3):
                offset = i * 27 + j * 3
                propagators.append(
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
        self.set_propagators(propagators)

    def pretty_print(self, solution: List[int]) -> None:
        for i in range(0, 81, 9):
            print(solution[i : i + 9])
