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
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_COUNT_EQ, ALG_SUM_EQ_C, ALG_LINEAR_EQ_C


class MagicSequenceProblem(Problem):
    """
    Find a sequence x_0, ... x_n-1 such that each x_i is the number of occurrences of i in the sequence.

    CSPLIB problem #19 - https://www.csplib.org/Problems/prob019/
    """

    def __init__(self, n: int, model_r1: bool = True, model_r2: bool = True):
        """
        Initializes the problem.

        :param n: the size of the sequence
        :type n: int
        """
        super().__init__([(0, n - 1)] * n)
        for i in range(n):
            self.add_propagator(ALG_COUNT_EQ, list(range(n)) + [i], [i])
        # redundant constraints
        if model_r1:
            self.add_propagator(ALG_SUM_EQ_C, range(n), [n])
        if model_r2:
            self.add_propagator(ALG_LINEAR_EQ_C, range(n), range(n + 1))
