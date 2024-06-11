import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.alldifferent import Alldifferent


class QueensProblem(Problem):

    def __init__(self, n: int):
        self.n = n
        shr_domains = np.array([[0, n - 1] * n]).reshape(n, 2)
        dom_indices = list(range(0, n)) * 3
        dom_offsets = [0] * n + list(range(0, n)) + list(range(0, -n, -1))
        propagators = [
            Alldifferent(list(range(0, n))),
            Alldifferent(list(range(n, 2 * n))),
            Alldifferent(list(range(2 * n, 3 * n))),
        ]
        super().__init__(shr_domains, dom_indices, dom_offsets, propagators)
