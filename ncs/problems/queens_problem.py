import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.alldifferent_puget_n2 import AlldifferentPugetN2


class QueensProblem(Problem):

    def __init__(self, n: int):
        self.n = n
        shr_domains = np.array([[0, n - 1] * n]).reshape(n, 2)
        dom_indices = list(range(0, n)) * 3
        dom_offsets = [0] * n + list(range(0, n)) + list(range(0, -n, -1))
        propagators = [
            AlldifferentPugetN2(list(range(0, n))),
            AlldifferentPugetN2(list(range(n, 2 * n))),
            AlldifferentPugetN2(list(range(2 * n, 3 * n))),
        ]
        super().__init__(shr_domains, dom_indices, dom_offsets, propagators)
