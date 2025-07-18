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
from nucs.problems.permutation_problem import PermutationProblem
from nucs.propagators.propagators import ALG_NO_SUB_CYCLE


class CircuitProblem(PermutationProblem):
    """
    A model for circuits.
    """

    def __init__(self, n: int):
        """
        Initializes the circuit problem.
        :param n: the number of vertices
        """
        self.n = n
        super().__init__(n)
        self.domains[0] = [1, n - 1]
        self.domains[n - 1] = [0, n - 2]
        self.domains[n] = [1, n - 1]
        self.domains[2 * n - 1] = [0, n - 2]
        self.add_propagator((list(range(n)), ALG_NO_SUB_CYCLE, []))
        self.add_propagator((list(range(n, 2 * n)), ALG_NO_SUB_CYCLE, []))
        # self.add_propagator((list(range(n)), ALG_SCC, []))  # not worth the cost
        # self.add_propagator((list(range(n, 2*n)), ALG_SCC, []))  # not worth the cost
