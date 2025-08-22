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
from nucs.propagators.propagators import ALG_ABS_EQ, ALG_ALLDIFFERENT, ALG_LEQ, ALG_SUM_EQ


class AllIntervalSeriesProblem(Problem):
    """
    CSPLIB problem #7 - https://www.csplib.org/Problems/prob007/
    """

    def __init__(self, n: int, symmetry_breaking: bool):
        """
        Initializes the problem.
        :param n: the size of the sequence
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        """
        super().__init__([(0, n - 1)] * n + [(-n + 1, n - 1)] * (n - 1) + [(1, n - 1)] * (n - 1))
        self.n = n
        for i in range(n - 1):
            self.add_propagator(ALG_SUM_EQ, [n + i, i, i + 1])
            self.add_propagator(ALG_ABS_EQ, [n + i, 2 * n - 1 + i])
        self.add_propagator(ALG_ALLDIFFERENT, range(n))
        self.add_propagator(ALG_ALLDIFFERENT, range(2 * n - 1, 3 * n - 2))
        if symmetry_breaking:
            self.add_propagator(ALG_LEQ, [0, 1], [-1])
            self.add_propagator(ALG_LEQ, [3 * n - 3, 2 * n - 1], [-1])

    def solution_as_printable(self, solution: NDArray) -> Any:
        return solution[: self.n].tolist()
