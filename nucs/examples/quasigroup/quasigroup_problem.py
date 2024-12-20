###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024 - Yan Georget
###############################################################################
from nucs.problems.latin_square_problem import M_COLOR, M_COLUMN, M_ROW, LatinSquareRCProblem
from nucs.propagators.propagators import ALG_ELEMENT_LIV


class QuasigroupProblem(LatinSquareRCProblem):
    """
    CSPLIB problem #3 - https://www.csplib.org/Problems/prob003/
    """

    def __init__(self, n: int, symmetry_breaking: bool = True):
        """
        Inits the problem.
        :param n: the size of the quasigroup
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        """
        super().__init__(n)
        # idempotence
        for model in [M_COLOR, M_ROW, M_COLUMN]:
            for i in range(n):
                self.shr_domains_lst[self.cell(i, i, model)] = [i, i]
        if symmetry_breaking:
            for i in range(1, n):
                self.shr_domains_lst[self.cell(i, n - 1)] = [i - 1, n - 1]


class Quasigroup5Problem(QuasigroupProblem):
    """
    Defined by:
       ((b∗a)∗b)∗b=a
    Can be enforced by:
       color[color[j, i] ,j] = row[i, j]
    which avoids the creation of additional variables
    """

    def __init__(self, n: int, symmetry_breaking: bool = True):
        """
        Inits the problem
        :param n: the size of the quasigroup
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        """
        super().__init__(n, symmetry_breaking)
        for j in range(n):
            for i in range(n):
                if i != j:
                    self.add_propagator(
                        (
                            [*self.column(j, M_COLOR), self.cell(j, i, M_COLOR), self.cell(i, j, M_ROW)],
                            ALG_ELEMENT_LIV,
                            [],
                        )
                    )
