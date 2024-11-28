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
from typing import List

from nucs.problems.circuit_problem import CircuitProblem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ELEMENT_IV


class TSPProblem(CircuitProblem):
    """ """

    def __init__(self, weights: List[List[int]]) -> None:
        """
        Inits the problem.
        :param weights: the weights between vertices
        """
        n = len(weights)
        super().__init__(n)
        max_weights = [max(weights_row) for weights_row in weights]
        self.add_variables([(0, max_weights[i]) for i in range(n)])
        self.add_variable((0, sum(max_weights)))
        for i in range(n):
            self.add_propagator(([i, 2 * n - 2 + i], ALG_ELEMENT_IV, weights[i]))
        self.add_propagator((list(range(2 * n - 2, 3 * n - 2)) + [3 * n - 2], ALG_AFFINE_EQ, [1] * n + [-1, 0]))
