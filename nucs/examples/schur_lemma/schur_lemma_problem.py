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
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_COUNT_EQ_C, ALG_COUNT_LEQ_C, ALG_LEXICOGRAPHIC_LEQ


class SchurLemmaProblem(Problem):
    """
    CSPLIB problem #15 - https://www.csplib.org/Problems/prob015/
    """

    def __init__(self, n: int, symmetry_breaking: bool = True) -> None:
        """
        Inits the problem.
        :param n: the size of the problem
        :param symmetry_breaking: a boolean indicating if symmetry constraints should be added to the model
        """
        super().__init__([(0, 1)] * n * 3)
        for x in range(n):
            self.add_propagator(([x * 3, x * 3 + 1, x * 3 + 2], ALG_COUNT_EQ_C, [1, 1]))
        for k in range(3):
            for x in range(n):
                for y in range(n):
                    z = (x + 1) + (y + 1) - 1
                    if 0 <= z < n:
                        self.add_propagator(([3 * x + k, 3 * y + k, 3 * z + k], ALG_COUNT_LEQ_C, [1, 2]))
        if symmetry_breaking:
            self.add_propagator(
                (
                    list(range(0, n * 3, 3)) + list(range(1, n * 3, 3)) + list(range(2, n * 3, 3)),
                    ALG_LEXICOGRAPHIC_LEQ,
                    [],
                )
            )
