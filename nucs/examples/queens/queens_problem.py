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
from typing import List

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
        Inits the problem.
        :param n: the number of queens
        """
        super().__init__([(0, n - 1)] * n, list(range(n)) * 3, [0] * n + list(range(n)) + list(range(0, -n, -1)))
        self.n = n
        self.add_propagators([(list(range(i * n, i * n + n)), ALG_ALLDIFFERENT, []) for i in range(3)])

    def solution_as_printable(self, solution: NDArray) -> List[List[str]]:
        solution_as_list = solution[: self.n].tolist()
        return [([" "] * i + ["Q"] + [" "] * (self.n - i - 1)) for i in solution_as_list]


class QueensDualProblem(Problem):
    """
    A dual model for the n-queens problem.

    CSPLIB problem #54 - https://www.csplib.org/Problems/prob054/
    """

    def __init__(self, n: int):
        """
        Inits the problem.
        :param n: the number of queens
        """
        super().__init__(
            [(0, n - 1)] * 2 * n,
            list(range(n)) * 3 + list(range(n, 2 * n)) * 3,
            [0] * n + list(range(n)) + list(range(0, -n, -1)) + [0] * n + list(range(n)) + list(range(0, -n, -1)),
        )
        self.n = n
        self.add_propagators([(list(range(i * n, i * n + n)), ALG_ALLDIFFERENT, []) for i in range(6)])
        self.add_propagator((list(range(n)) + list(range(3 * n, 4 * n)), ALG_PERMUTATION_AUX, []))

    def solution_as_printable(self, solution: NDArray) -> List[List[str]]:
        solution_as_list = solution[: self.n].tolist()
        return [([" "] * i + ["Q"] + [" "] * (self.n - i - 1)) for i in solution_as_list]
