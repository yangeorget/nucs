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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from nucs.problems.latin_square_problem import M_COLOR, M_COLUMN, M_ROW, LatinSquareProblem, LatinSquareRCProblem
from nucs.propagators.propagators import ALG_ELEMENT_LIC, ALG_ELEMENT_LIV, ALG_ELEMENT_LIV_ALLDIFFERENT


class QuasigroupProblem(LatinSquareProblem):
    """
    CSPLIB problem #3 - https://www.csplib.org/Problems/prob003/
    """

    def __init__(self, kind: int, n: int, idempotent: bool, symmetry_breaking: bool = True):
        """
        Inits the problem.
        :param n: the size of the quasigroup
        :param idempotent: a boolean indicating if the quasigroup is idempotent
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        """
        super().__init__(list(range(n)))
        if symmetry_breaking:
            for i in range(1, n):
                self.shr_domains_lst[self.cell(i, n - 1)] = [i - 1, n - 1]
        if idempotent:
            for i in range(n):
                self.shr_domains_lst[self.cell(i, i)] = [i, i]
        if kind == 3:  # (a∗b)∗(b*a)=a
            # color[color[i, j], color[j, i]] = i
            pass
        elif kind == 4:  # (b∗a)∗(a*b) = a
            # color[color[j, i], color[i, j]] = i
            pass
        elif kind == 5:  # ((b∗a)∗b)∗b = a
            # color[color[color[j, i], j], j] = i
            for i in range(n):
                for j in range(n):
                    if not idempotent or j != i:
                        jij = self.add_variable((0, n - 1))
                        self.add_propagator(([*self.column(j), self.cell(j, i), jij], ALG_ELEMENT_LIV, []))
                        self.add_propagator(([*self.column(j), jij], ALG_ELEMENT_LIC, [i]))
        elif kind == 6:  # (a∗b)∗b = a*(a*b)
            # color[color[i, j], j] = color[i, color[i, j]]
            pass
        elif kind == 7:  # (b∗a)∗b = a*(b*a)
            # color[color[j, i], j] = color[i, color[j, i]]
            pass


class QuasigroupProblemRC(LatinSquareRCProblem):
    """
    CSPLIB problem #3 - https://www.csplib.org/Problems/prob003/
    """

    def __init__(self, kind: int, n: int, idempotent: bool, symmetry_breaking: bool = True):
        """
        Inits the problem.
        :param n: the size of the quasigroup
        :param idempotent: a boolean indicating if the quasigroup is idempotent
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        """
        super().__init__(n)
        if symmetry_breaking:
            for i in range(1, n):
                self.shr_domains_lst[self.cell(i, n - 1)] = [i - 1, n - 1]
        if idempotent:
            for model in [M_COLOR, M_ROW, M_COLUMN]:
                for i in range(n):
                    self.shr_domains_lst[self.cell(i, i, model)] = [i, i]
                for i in range(1, n):
                    self.shr_domains_lst[self.cell(0, i, model)] = [1, n - 1]
                for i in range(0, n - 1):
                    self.shr_domains_lst[self.cell(n - 1, i, model)] = [0, n - 2]
        for i in range(n):
            for j in range(n):
                if not idempotent or j != i:
                    if kind == 3:  # (a∗b)∗(b*a)=a
                        # color[color[i, j], color[j, i]] = i
                        # column[color[i, j], i] = color[j, i] which avoids the creation of additional variables
                        self.add_propagator(
                            (
                                [*self.column(i, M_COLUMN), self.cell(i, j, M_COLOR), self.cell(j, i, M_COLOR)],
                                ALG_ELEMENT_LIV_ALLDIFFERENT,
                                [],
                            )
                        )
                    elif kind == 4:  # (b∗a)∗(a*b) = a
                        # color[color[j, i], color[i, j]] = i
                        # column[color[j, i], i] = color[i, j] which avoids the creation of additional variables
                        self.add_propagator(
                            (
                                [*self.column(i, M_COLUMN), self.cell(j, i, M_COLOR), self.cell(i, j, M_COLOR)],
                                ALG_ELEMENT_LIV_ALLDIFFERENT,
                                [],
                            )
                        )
                    elif kind == 5:  # ((b∗a)∗b)∗b = a
                        # color[color[color[j, i], j], j] = i
                        # row[i, j] = color[color[j, i], j] which avoids the creation of additional variables
                        self.add_propagator(
                            (
                                [*self.column(j, M_COLOR), self.cell(j, i, M_COLOR), self.cell(i, j, M_ROW)],
                                ALG_ELEMENT_LIV_ALLDIFFERENT,
                                [],
                            )
                        )
                    elif kind == 6:  # (a∗b)∗b = a*(a*b)
                        # color[color[i, j], j] = color[i, color[i, j]]
                        ijj = self.add_variable((0, n - 1))
                        self.add_propagator(
                            (
                                [*self.column(j, M_COLOR), self.cell(i, j, M_COLOR), ijj],
                                ALG_ELEMENT_LIV_ALLDIFFERENT,
                                [],
                            )
                        )
                        self.add_propagator(
                            ([*self.row(i, M_COLOR), self.cell(i, j, M_COLOR), ijj], ALG_ELEMENT_LIV_ALLDIFFERENT, [])
                        )
                    elif kind == 7:  # (b∗a)∗b = a*(b*a)
                        # color[color[j, i], j] = color[i, color[j, i]]
                        jij = self.add_variable((0, n - 1))
                        self.add_propagator(
                            (
                                [*self.column(j, M_COLOR), self.cell(j, i, M_COLOR), jij],
                                ALG_ELEMENT_LIV_ALLDIFFERENT,
                                [],
                            )
                        )
                        self.add_propagator(
                            ([*self.row(i, M_COLOR), self.cell(j, i, M_COLOR), jij], ALG_ELEMENT_LIV_ALLDIFFERENT, [])
                        )
