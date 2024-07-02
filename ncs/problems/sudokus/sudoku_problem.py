from typing import List

import numpy as np

from ncs.problems.problem import ALLDIFFERENT_LOPEZ_ORTIZ, Problem
from ncs.propagators.propagator import Propagator


class SudokuProblem(Problem):

    def __init__(self, givens: List[List[int]]):
        super().__init__(
            shr_domains=[(1, 9) if given == 0 else (given, given) for line in givens for given in line],
            dom_indices=list(range(0, 81)),
            dom_offsets=[0] * 81,
        )
        propagators = []
        for i in range(0, 9):
            propagators.append(
                Propagator(np.array(list(range(0 + i * 9, 9 + i * 9)), dtype=np.int32), ALLDIFFERENT_LOPEZ_ORTIZ)
            )
            propagators.append(
                Propagator(np.array(list(range(0 + i, 81 + i, 9)), dtype=np.int32), ALLDIFFERENT_LOPEZ_ORTIZ)
            )
        for i in range(0, 3):
            for j in range(0, 3):
                propagators.append(
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
        self.set_propagators(propagators)

    def pretty_print(self, solution: List[int]) -> None:
        for i in range(0, 81, 9):
            print(solution[i : i + 9])
