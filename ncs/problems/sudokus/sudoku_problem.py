import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import Problem
from ncs.propagators.alldifferent import Alldifferent


class SudokuProblem(Problem):

    def __init__(self, givens: NDArray):
        super().__init__(
            np.array(self.build_domains(givens)).reshape(9, 2),
            #list(range(0, 81)),
            #list(range(0, 81)),
        )
        for i in range(0, 9):
            self.add_propagator(Alldifferent(list(range(0 + i * 9, 9 + i * 9))))
            self.add_propagator(Alldifferent(list(range(0 + i, 72 + i, 9))))
        for i in range(0, 3):
            for j in range(0, 3):
                self.add_propagator(
                    Alldifferent(
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
                        ]
                    )
                )

    def build_domains(self, givens: NDArray):
        pass
