import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.alldifferent import Alldifferent


class QueensProblem(Problem):

    def __init__(self, n: int):
        super().__init__(
            np.array([[0, n - 1] * n]).reshape(n, 2), # TODO: move ndarray
            list(range(0, n)) * 3,
            [0] * n + list(range(0, n)) + list(range(0, -n, -1)),
        )
        self.n = n
        self.add_propagator(Alldifferent(list(range(0, n))))
        self.add_propagator(Alldifferent(list(range(n, 2 * n))))
        self.add_propagator(Alldifferent(list(range(2 * n, 3 * n))))
