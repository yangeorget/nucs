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
                Propagator(np.array(list(range(0, n)), dtype=np.int32), ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ),
                Propagator(np.array(list(range(n, 2 * n)), dtype=np.int32), ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ),
                Propagator(np.array(list(range(2 * n, 3 * n)), dtype=np.int32), ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ),
            ]
        )
