import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.alldifferent_puget_n2 import AlldifferentPugetN2
from ncs.propagators.shift import Shift


class QueensProblem(Problem):

    def __init__(self, n: int):
        self.n = n
        x = list(range(0, n))
        xp = list(range(n, 2 * n))
        xm = list(range(2 * n, 3 * n))
        super().__init__(
            np.array([([0, n - 1] * n) + ([0, 2 * n - 2] * n) + ([-n + 1, n - 1] * n)]).reshape(3 * n, 2),
            [
                AlldifferentPugetN2(x),
                AlldifferentPugetN2(xp),
                AlldifferentPugetN2(xm),
                Shift(xp + x, list(range(0, n))),
                Shift(x + xm, list(range(0, n))),
            ],
        )
