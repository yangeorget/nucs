from nucs.problems.latin_square_problem import LatinSquareRCProblem


class QuasigroupProblem(LatinSquareRCProblem):
    """
    This is problem #3 on CSPLIB (https://www.csplib.org/Problems/prob003/).
    """

    def __init__(self, n: int):
        super().__init__(n)
        for i in range(n):
            self.shr_domains_list[i + i * n] = i  # idempotence
        # TODO: symmetry breaking


class Quasigroup5Problem(QuasigroupProblem):
    """
    ((b∗a)∗b)∗b=a
    """

    def __init__(self, n: int):
        super().__init__(n)
        # TODO:
        # add variables
        # add specific constraints
