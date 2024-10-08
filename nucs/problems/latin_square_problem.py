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
from typing import List, Optional

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_ELEMENT_LIC

M_COLOR = 0  # the color model
M_ROW = 1  # the row model
M_COLUMN = 2  # the column model


class LatinSquareProblem(Problem):
    """
    A simple model for latin squares.
    """

    def __init__(self, colors: List[int], givens: Optional[List[List[int]]] = None):
        """
        Inits the latin square.
        :colors: the possible values for the cells,
        usually [0, ..., n-1] except in some cases (eg Sudokus) where [1, ..., n] is preferred;
        the number of colors is also the size of the square
        :givens: initial values for the cells, any value different from the possible colors is used as a wildcard
        """
        self.colors = colors
        self.n = len(colors)
        if givens is None:
            shr_domains = [(colors[0], colors[-1])] * self.n**2
        else:
            shr_domains = [
                (colors[0], colors[-1]) if given not in self.colors else (given, given)
                for line in givens
                for given in line
            ]
        super().__init__(shr_domains)
        self.add_propagators([(self.row(i), ALG_ALLDIFFERENT, []) for i in range(self.n)])
        self.add_propagators([(self.column(j), ALG_ALLDIFFERENT, []) for j in range(self.n)])

    def cell(self, i: int, j: int, model: int = M_COLOR) -> int:
        offset = model * (self.n**2)
        return offset + i * self.n + j

    def row(self, i: int, model: int = M_COLOR) -> List[int]:
        offset = model * (self.n**2)
        return list(range(offset + i * self.n, offset + self.n + i * self.n))

    def column(self, j: int, model: int = M_COLOR) -> List[int]:
        offset = model * (self.n**2)
        return list(range(offset + j, offset + self.n**2 + j, self.n))

    def solution_as_matrix(self, solution: List[int]) -> List[List[int]]:
        return [solution[i : i + self.n] for i in range(0, self.n**2, self.n)]


class LatinSquareRCProblem(LatinSquareProblem):
    """
    A full model for latin squares with 3 kinds of variables:
    - n*n variables for the values (aka colors), these variables are indexed by rows then columns
    - n*n variables for the rows, these variables are indexed by colors then columns
    - n*n variables for the columns, these variables are indexed by rows then colors
    """

    def __init__(self, n: int):
        """
        Inits the problem.
        :param n: the size of the square
        """
        super().__init__(list(range(n)))  # the color model
        self.add_variables([(0, n - 1)] * n**2)  # the row model
        self.add_variables([(0, n - 1)] * n**2)  # the column model
        self.add_propagators([(self.row(i, M_ROW), ALG_ALLDIFFERENT, []) for i in range(self.n)])
        self.add_propagators([(self.column(j, M_ROW), ALG_ALLDIFFERENT, []) for j in range(self.n)])
        self.add_propagators([(self.row(i, M_COLUMN), ALG_ALLDIFFERENT, []) for i in range(self.n)])
        self.add_propagators([(self.column(j, M_COLUMN), ALG_ALLDIFFERENT, []) for j in range(self.n)])
        # row[c,j]=i <=> column[i,c]=j
        for c in range(n):
            for i in range(n):
                self.add_propagator(([*self.row(c, M_ROW), self.cell(i, c, M_COLUMN)], ALG_ELEMENT_LIC, [i]))
            for j in range(n):
                self.add_propagator(([*self.column(c, M_COLUMN), self.cell(c, j, M_ROW)], ALG_ELEMENT_LIC, [j]))
        # row[c,j]=i <=> color[i,j]=c
        for j in range(n):
            for i in range(n):
                self.add_propagator(([*self.column(j, M_ROW), self.cell(i, j, M_COLOR)], ALG_ELEMENT_LIC, [i]))
            for c in range(n):
                self.add_propagator(([*self.column(j, M_COLOR), self.cell(c, j, M_ROW)], ALG_ELEMENT_LIC, [c]))
        # color[i,j]=c <=> column[i,c]=j
        for i in range(n):
            for c in range(n):
                self.add_propagator(([*self.row(i, M_COLOR), self.cell(i, c, M_COLUMN)], ALG_ELEMENT_LIC, [c]))
            for j in range(n):
                self.add_propagator(([*self.row(i, M_COLUMN), self.cell(i, j, M_COLOR)], ALG_ELEMENT_LIC, [j]))
