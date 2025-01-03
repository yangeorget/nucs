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
from nucs.problems.latin_square_problem import M_COLOR, M_COLUMN, M_ROW, LatinSquareRCProblem
from nucs.propagators.propagators import ALG_ELEMENT_LIV


class QuasigroupProblem(LatinSquareRCProblem):
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
        if kind == 3:  # (a∗b)∗(b*a)=a
            # Equivalent to: color[color[i, j], color[j, i]] = i
            # Equivalent to: column[color[i, j], i] = color[j, i] which avoids the creation of additional variables
            self.add_propagators(
                [
                    (
                        [*self.column(i, M_COLUMN), self.cell(i, j, M_COLOR), self.cell(j, i, M_COLOR)],
                        ALG_ELEMENT_LIV,
                        [],
                    )
                    for i in range(n)
                    for j in range(n)
                ]
            )
        elif kind == 4:  # (b∗a)∗(a*b) = a
            # Equivalent to: color[color[j, i], color[i, j]] = i
            # Equivalent to: column[color[j, i], i] = color[i, j] which avoids the creation of additional variables
            self.add_propagators(
                [
                    (
                        [*self.column(i, M_COLUMN), self.cell(j, i, M_COLOR), self.cell(i, j, M_COLOR)],
                        ALG_ELEMENT_LIV,
                        [],
                    )
                    for i in range(n)
                    for j in range(n)
                ]
            )
        elif kind == 5:  # ((b∗a)∗b)∗b = a
            # Equivalent to: color[color[color[j, i], j], j] = i
            # Equivalent to: row[i, j] = color[color[j, i], j] which avoids the creation of additional variables
            self.add_propagators(
                [
                    (
                        [*self.column(j, M_COLOR), self.cell(j, i, M_COLOR), self.cell(i, j, M_ROW)],
                        ALG_ELEMENT_LIV,
                        [],
                    )
                    for i in range(n)
                    for j in range(n)
                ]
            )
        elif kind == 6:  # (a∗b)∗b = a*(a*b)
            # Equivalent to: color[color[i, j], j] = color[i, color[i, j]]
            additional_vars_idx = self.add_variables([(0, n - 1)] * n**2)  # additional variables
            for i in range(n):
                for j in range(n):
                    self.add_propagator(
                        (
                            [*self.column(j, M_COLOR), self.cell(i, j, M_COLOR), additional_vars_idx + i * n + j],
                            ALG_ELEMENT_LIV,
                            [],
                        )
                    )
                    self.add_propagator(
                        (
                            [*self.row(i, M_COLOR), self.cell(i, j, M_COLOR), additional_vars_idx + i * n + j],
                            ALG_ELEMENT_LIV,
                            [],
                        )
                    )
        elif kind == 7:  # (b∗a)∗b = a*(b*a)
            # Equivalent to: color[color[j, i], j] = color[i, color[j, i]]
            additional_vars_idx = self.add_variables([(0, n - 1)] * n**2)  # additional variables
            for i in range(n):
                for j in range(n):
                    self.add_propagator(
                        (
                            [*self.column(j, M_COLOR), self.cell(j, i, M_COLOR), additional_vars_idx + i * n + j],
                            ALG_ELEMENT_LIV,
                            [],
                        )
                    )
                    self.add_propagator(
                        (
                            [*self.row(i, M_COLOR), self.cell(j, i, M_COLOR), additional_vars_idx + i * n + j],
                            ALG_ELEMENT_LIV,
                            [],
                        )
                    )
        if idempotent:
            for model in [M_COLOR, M_ROW, M_COLUMN]:
                for i in range(n):
                    self.shr_domains_lst[self.cell(i, i, model)] = [i, i]
        if symmetry_breaking:
            for i in range(1, n):
                self.shr_domains_lst[self.cell(i, n - 1)] = [i - 1, n - 1]
