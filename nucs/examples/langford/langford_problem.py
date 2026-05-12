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
# Copyright 2024-2026 - Yan Georget
###############################################################################
from typing import Any

from numpy.typing import NDArray

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_ADD_C_EQ


class LangfordProblem(Problem):
    """
    CSPLIB problem #24 - https://www.csplib.org/Problems/prob024/
    """

    def index(self, i: int, j: int) -> int:
        return i * self.n + j

    def color(self, idx: int) -> int:
        return idx % self.n

    def __init__(self, k: int, n: int) -> None:
        """
        Initializes the problem.

        :param k: the number of occurences
        :type k: int
        :param n: the number of values
        :type n: int
        """
        self.k = k
        self.n = n
        domains = [(0, k * n - 1)] * k * n  # domain[index(i,j)] is the position of the ith occurence of j
        super().__init__(domains)
        self.add_propagator(ALG_ALLDIFFERENT, range(0, k * n))
        for i in range(k - 1):
            for j in range(n):
                self.add_propagator(ALG_ADD_C_EQ, [self.index(i, j), self.index(i + 1, j)], [j + 2])

    def solution_as_printable(self, solution: NDArray) -> Any:
        positions = solution.tolist()
        colors = [0] * self.n * self.k
        for idx in range(self.k * self.n):
            colors[positions[idx]] = idx % self.n
        return colors
