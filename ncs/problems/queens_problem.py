from ncs.problems.problem import Problem
from ncs.propagators.propagators import ALG_ALLDIFFERENT


class QueensProblem(Problem):
    """
    A simple model for the n-queens problem.
    """

    def __init__(self, n: int):
        super().__init__(
            shr_domains=[(0, n - 1)] * n,
            dom_indices=list(range(n)) * 3,
            dom_offsets=[0] * n + list(range(n)) + list(range(0, -n, -1)),
        )
        self.set_propagators(
            [
                (list(range(n)), ALG_ALLDIFFERENT, []),
                (list(range(n, 2 * n)), ALG_ALLDIFFERENT, []),
                (list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []),
            ]
        )
