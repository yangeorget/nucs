import numpy as np

from ncs.problems.problem import ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ, Problem
from ncs.propagators.propagator import Propagator


class QueensProblem(Problem):

    def __init__(self, n: int):
        super().__init__(
            shared_domains=[(0, n - 1)] * n,
            domain_indices=list(range(0, n)) * 3,
            domain_offsets=[0] * n + list(range(0, n)) + list(range(0, -n, -1)),
        )
        self.set_propagators(
            [
                (list(range(0, n)), ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ),
                (list(range(n, 2 * n)), ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ),
                (list(range(2 * n, 3 * n)), ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ),
            ]
        )
