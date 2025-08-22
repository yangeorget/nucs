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
from typing import Any, Iterable, List, Optional

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_PERMUTATION_AUX

M_COLOR = 0  # the color model
M_ROW = 1  # the row model
M_COLUMN = 2  # the column model


class LatinSquareProblem(Problem):
    """
    A simple model for latin squares.
    """

    def __init__(self, colors: Iterable[int], givens: Optional[List[List[int]]] = None):
        """
        Initializes the latin square.
        :colors: the possible values for the cells,
        usually [0, ..., n-1] except in some cases (e.g., Sudokus) where [1, ..., n] is preferred;
        the number of colors is also the size of the square
        :givens: initial values for the cells, any value different from the possible colors is used as a wildcard
        """
        self.colors = list(colors)
        self.n = len(self.colors)
        if givens is None:
            shr_domains = [(self.colors[0], self.colors[-1])] * self.n**2
        else:
            shr_domains = [
                (self.colors[0], self.colors[-1]) if given not in self.colors else (given, given)
                for line in givens
                for given in line
            ]
        super().__init__(shr_domains)
        for i in range(self.n):
            self.add_propagator(ALG_ALLDIFFERENT, self.row(i))
            self.add_propagator(ALG_ALLDIFFERENT, self.column(i))

    def cell(self, i: int, j: int, model: int = M_COLOR) -> int:
        offset = model * (self.n**2)
        return offset + i * self.n + j

    def row(self, i: int, model: int = M_COLOR) -> Iterable[int]:
        offset = model * (self.n**2)
        return range(offset + i * self.n, offset + self.n + i * self.n)

    def column(self, j: int, model: int = M_COLOR) -> Iterable[int]:
        offset = model * (self.n**2)
        return range(offset + j, offset + self.n**2 + j, self.n)

    def solution_as_printable(self, solution: NDArray) -> Any:
        solution_as_list = solution.tolist()
        return [solution_as_list[i : i + self.n] for i in range(0, self.n**2, self.n)]


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
        super().__init__(range(n))  # the color model
        self.add_variables([(0, n - 1)] * n**2)  # the row model
        self.add_variables([(0, n - 1)] * n**2)  # the column model
        for i in range(self.n):
            self.add_propagator(ALG_ALLDIFFERENT, self.row(i, M_ROW))
            self.add_propagator(ALG_ALLDIFFERENT, self.column(i, M_ROW))
            self.add_propagator(ALG_ALLDIFFERENT, self.row(i, M_COLUMN))
            self.add_propagator(ALG_ALLDIFFERENT, self.column(i, M_COLUMN))
        # row[c,j]=i <=> color[i,j]=c
        for j in range(n):
            self.add_propagator(ALG_PERMUTATION_AUX, [*self.column(j), *self.column(j, M_ROW)])
        # row[c,j]=i <=> column[i,c]=j
        for c in range(n):
            self.add_propagator(ALG_PERMUTATION_AUX, [*self.row(c, M_ROW), *self.column(c, M_COLUMN)])
        # color[i,j]=c <=> column[i,c]=j
        for i in range(n):
            self.add_propagator(ALG_PERMUTATION_AUX, [*self.row(i), *self.row(i, M_COLUMN)])
