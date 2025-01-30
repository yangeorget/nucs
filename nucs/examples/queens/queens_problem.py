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
from nucs.propagators.propagators import ALG_ALLDIFFERENT


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
        super().__init__(
            [(0, n - 1)] * n,
            list(range(n)) * 3,
            [0] * n + list(range(n)) + list(range(0, -n, -1)),
        )
        self.n = n
        self.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
        self.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
        self.add_propagator((list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []))

    def solution_as_printable(self, solution: NDArray) -> List[List[str]]:
        solution_as_list = solution[: self.n].tolist()
        return [([" "] * i + ["X"] + [" "] * (self.n - i - 1)) for i in solution_as_list]
