from nucs.problems.problem import Problem


class QuasigroupProblem(Problem):
    """
    This is problem #3 on CSPLIB (https://www.csplib.org/Problems/prob003/).
    """

    def __init__(self, n: int):
        super().__init__([(0, n - 1)] * n**2)
