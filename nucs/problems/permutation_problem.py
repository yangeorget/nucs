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
from nucs.propagators.propagators import ALG_ALLDIFFERENT, ALG_PERMUTATION_AUX


class PermutationProblem(Problem):
    """
    A model for permutation.
    """

    def __init__(self, n: int):
        """
        Inits the permutation problem.
        :param n: the number variables/values
        """
        self.n = n
        super().__init__([(0, n - 1)] * 2 * n)
        self.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
        self.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
        self.add_propagator((list(range(2 * n)), ALG_PERMUTATION_AUX, []))
