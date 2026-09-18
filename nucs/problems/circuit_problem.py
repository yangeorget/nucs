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
from nucs.problems.permutation_problem import PermutationProblem
from nucs.propagators.propagators import ALG_CIRCUIT_CHAINS


class CircuitProblem(PermutationProblem):
    """
    A model for circuits: the successors and the predecessors are each a permutation forming a single circuit,
    enforced by CIRCUIT_CHAINS.
    """

    def __init__(self, n: int):
        """
        Initializes the circuit problem.

        :param n: the number of vertices
        :type n: int
        """
        self.n = n
        super().__init__(n)
        self.domains[0] = (1, n - 1)
        self.domains[n - 1] = (0, n - 2)
        self.domains[n] = (1, n - 1)
        self.domains[2 * n - 1] = (0, n - 2)
        self.add_propagator(ALG_CIRCUIT_CHAINS, range(n), [0])
        self.add_propagator(ALG_CIRCUIT_CHAINS, range(n, 2 * n), [0])
