from nucs.problems.latin_square_problem import LatinSquareProblem


class QuasigroupProblem(LatinSquareProblem):
    """
    This is problem #3 on CSPLIB (https://www.csplib.org/Problems/prob003/).
    """

    def __init__(self, n: int):
        super().__init__(list(range(0, n)))
        for i in range(n):
            self.shr_domains_list[i + i * n] = i  # idempotence
        # TODO: symmetry breaking


class Quasigroup5Problem(QuasigroupProblem):
    def __init__(self, n: int):
        super().__init__(n)
        # add variables
        # add specific constraints
