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
# Copyright 2024 - Yan Georget
###############################################################################
from nucs.problems.permutation_problem import PermutationProblem
from nucs.propagators.propagators import ALG_NO_SUB_CYCLE


class CircuitProblem(PermutationProblem):
    """
    A model for circuit.
    """

    def __init__(self, n: int):
        """
        Inits the circuit problem.
        :param n: the number of vertices
        """
        self.n = n
        super().__init__(n)  # TODO: do we want to specify the domains more precisely ?
        self.add_propagator((list(range(n)), ALG_NO_SUB_CYCLE, []))
        self.add_propagator((list(range(n, 2 * n)), ALG_NO_SUB_CYCLE, []))
        # self.add_propagator((list(range(n)), ALG_SCC, []))  # not worth the cost
        # self.add_propagator((list(range(n, 2*n)), ALG_SCC, []))  # not worth the cost
