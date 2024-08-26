from typing import List

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_AFFINE_LEQ


class KnapsackProblem(Problem):
    """
    This is problem #133 on https://www.csplib.org/Problems/prob133/.
    """

    def __init__(self, weights: List[int], volumes: List[int], capacity: int) -> None:
        n = len(weights)
        super().__init__(
            shr_domains=[(0, 1)] * n + [(0, sum(weights))], dom_indices=list(range(n + 1)), dom_offsets=[0] * (n + 1)
        )
        self.set_propagators(
            [
                (list(range(n)), ALG_AFFINE_LEQ, [*volumes, capacity]),
                (list(range(n + 1)), ALG_AFFINE_EQ, [*weights, -1, 0]),
            ]
        )
        self.weight = n
