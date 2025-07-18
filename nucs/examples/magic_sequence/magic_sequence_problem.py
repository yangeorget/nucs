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
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_COUNT_EQ


class MagicSequenceProblem(Problem):
    """
    Find a sequence x_0, ... x_n-1 such that each x_i is the number of occurrences of i in the sequence.

    CSPLIB problem #19 - https://www.csplib.org/Problems/prob019/
    """

    def __init__(self, n: int):
        """
        Initializes the problem.
        :param n: the size of the sequence
        """
        super().__init__([(0, n)] * n)
        for i in range(n):
            self.add_propagator((list(range(n)) + [i], ALG_COUNT_EQ, [i]))
        # redundant constraints
        self.add_propagator((list(range(n)), ALG_AFFINE_EQ, [1] * n + [n]))
        self.add_propagator((list(range(n)), ALG_AFFINE_EQ, list(range(n)) + [n]))
