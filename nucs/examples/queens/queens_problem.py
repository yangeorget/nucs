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
from typing import Any

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_PERMUTATION_AUX


class QueensProblem(Problem):
    """
    A simple model for the n-queens problem.

    CSPLIB problem #54 - https://www.csplib.org/Problems/prob054/
    """

    def __init__(self, n: int):
        """
        Initializes the problem.
        :param n: the number of queens
        """
        super().__init__([(0, n - 1)] * n)
        self.n = n
        variables = range(0, n)
        self.add_propagator(ALG_ALLDIFFERENT, variables)
        self.add_propagator(ALG_ALLDIFFERENT, variables, range(n))
        self.add_propagator(ALG_ALLDIFFERENT, variables, range(0, -n, -1))

    def solution_as_printable(self, solution: NDArray) -> Any:
        return [([" "] * i + ["Q"] + [" "] * (self.n - i - 1)) for i in (solution[: self.n])]


class QueensDualProblem(Problem):
    """
    A dual model for the n-queens problem.

    CSPLIB problem #54 - https://www.csplib.org/Problems/prob054/
    """

    def __init__(self, n: int):
        """
        Initializes the problem.
        :param n: the number of queens
        """
        super().__init__([(0, n - 1)] * 2 * n)
        self.n = n
        column_variables = range(0, n)
        self.add_propagator(ALG_ALLDIFFERENT, column_variables)
        self.add_propagator(ALG_ALLDIFFERENT, column_variables, range(n))
        self.add_propagator(ALG_ALLDIFFERENT, column_variables, range(0, -n, -1))
        row_variables = range(n, 2 * n)
        self.add_propagator(ALG_ALLDIFFERENT, row_variables)
        self.add_propagator(ALG_ALLDIFFERENT, row_variables, range(n))
        self.add_propagator(ALG_ALLDIFFERENT, row_variables, range(0, -n, -1))
        self.add_propagator(ALG_PERMUTATION_AUX, list(column_variables) + list(row_variables))

    def solution_as_printable(self, solution: NDArray) -> Any:
        return [([" "] * i + ["Q"] + [" "] * (self.n - i - 1)) for i in (solution[: self.n])]
