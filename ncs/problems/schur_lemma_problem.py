from ncs.problems.problem import Problem
from ncs.propagators.propagators import (
    ALG_AFFINE_LEQ,
    ALG_EXACTLY_EQ,
    ALG_LEXICOGRAPHIC_LEQ,
)


class SchurLemmaProblem(Problem):
    """
    See https://www.csplib.org/Problems/prob015/.
    """

    def __init__(self, n: int) -> None:
        super().__init__(shr_domains=[(0, 1)] * n * 3, dom_indices=list(range(n * 3)), dom_offsets=[0] * n * 3)
        propagators = []
        for x in range(n):
            propagators.append(([x * 3, x * 3 + 1, x * 3 + 2], ALG_EXACTLY_EQ, [1, 1]))
        for x in range(n):
            for y in range(n):
                z = (x + 1) + (y + 1) - 1
                if 0 <= z < n:
                    for k in range(3):
                        propagators.append(([3 * x + k, 3 * y + k, 3 * z + k], ALG_AFFINE_LEQ, [1, 1, 1, 2]))
        # breaking symmetries
        propagators.append(
            (list(range(0, n * 3, 3)) + list(range(1, n * 3, 3)) + list(range(2, n * 3, 3)), ALG_LEXICOGRAPHIC_LEQ, [])
        )
        self.set_propagators(propagators)
