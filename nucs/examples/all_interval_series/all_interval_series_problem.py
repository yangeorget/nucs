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
from nucs.propagators.propagators import ALG_ABS_EQ, ALG_AFFINE_EQ, ALG_AFFINE_LEQ, ALG_ALLDIFFERENT


class AllIntervalSeriesProblem(Problem):
    """
    CSPLIB problem #7 - https://www.csplib.org/Problems/prob007/
    """

    def __init__(self, n: int, symmetry_breaking: bool):
        """
        Inits the problem.
        :param n: the size of the sequence
        """
        super().__init__([(0, n - 1)] * n + [(-n + 1, n - 1)] * (n - 1) + [(1, n - 1)] * (n - 1))
        self.n = n
        for i in range(n - 1):
            self.add_propagator(([n + i, i + 1, i], ALG_AFFINE_EQ, [1, -1, 1, 0]))
            self.add_propagator(([n + i, 2 * n - 1 + i], ALG_ABS_EQ, []))
        self.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
        self.add_propagator((list(range(2 * n - 1, 3 * n - 2)), ALG_ALLDIFFERENT, []))
        if symmetry_breaking:
            self.add_propagator(([0, 1], ALG_AFFINE_LEQ, [1, -1, -1]))
            self.add_propagator(([3 * n - 3, 2 * n - 1], ALG_AFFINE_LEQ, [1, -1, -1]))

    def solution_as_printable(self, solution: NDArray) -> Any:
        return solution[: self.n].tolist()
