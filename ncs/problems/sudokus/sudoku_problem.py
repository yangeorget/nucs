from typing import List

import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.propagator import ALLDIFFERENT_LOPEZ_ORTIZ, Propagator


class SudokuProblem(Problem):

    def __init__(self, givens: List[List[int]]):
        shr_domains = [(1, 9) if given == 0 else (given, given) for line in givens for given in line]
        indices = list(range(0, 81))
        offsets = [0] * 81
        super().__init__(shr_domains, indices, offsets)
        for i in range(0, 9):
            self.add_propagator(
                Propagator(np.array(list(range(0 + i * 9, 9 + i * 9)), dtype=np.int32), ALLDIFFERENT_LOPEZ_ORTIZ)
            )
            self.add_propagator(
                Propagator(np.array(list(range(0 + i, 81 + i, 9)), dtype=np.int32), ALLDIFFERENT_LOPEZ_ORTIZ)
            )
        for i in range(0, 3):
            for j in range(0, 3):
                self.add_propagator(
                    Propagator(
                        np.array(
                            [
                                0 + i * 27 + j * 3,
                                1 + i * 27 + j * 3,
                                2 + i * 27 + j * 3,
                                9 + i * 27 + j * 3,
                                10 + i * 27 + j * 3,
                                11 + i * 27 + j * 3,
                                18 + i * 27 + j * 3,
                                19 + i * 27 + j * 3,
                                20 + i * 27 + j * 3,
                            ],
                            dtype=np.int32,
                        ),
                        ALLDIFFERENT_LOPEZ_ORTIZ,
                    )
                )

    def pretty_print(self, solution: List[int]) -> None:
        for i in range(0, 81, 9):
            print(solution[i : i + 9])
