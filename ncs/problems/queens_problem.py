import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.alldifferent_puget_n3 import AlldifferentPugetN3
from ncs.propagators.shift import Shift


class QueensProblem(Problem):

    def __init__(self, n: int):
        self.n = n
        self.domains = np.array([([0, n - 1] * n) + ([0, 2 * n - 2] * n) + ([-n + 1, n - 1] * n)]).reshape(3 * n, 2)
        x = list(range(0, n))
        xp = list(range(n, 2 * n))
        xm = list(range(2 * n, 3 * n))
        self.propagators = [
            AlldifferentPugetN3(np.array(x)),
            AlldifferentPugetN3(np.array(xp)),
            AlldifferentPugetN3(np.array(xm)),
            Shift(np.array(xp + x).reshape(n, 2, order="F"), np.array(list(range(0, n)))),
            Shift(np.array(xm + x).reshape(n, 2, order="F"), -np.array(list(range(0, n)))),
        ]
