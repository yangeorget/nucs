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

    def __init__(self, cost_rows: List[List[int]]) -> None:
        """
        Inits the problem.
        :param cost_rows: the costs between vertices
        """
        n = len(cost_rows)
        super().__init__(n)
        max_costs = [max(cost_row) for cost_row in cost_rows]
        min_costs = [min([cost for cost in cost_row if cost > 0]) for cost_row in cost_rows]
        self.add_variables([(min_costs[i], max_costs[i]) for i in range(n)])  # the costs
        self.add_variable((sum(min_costs), sum(max_costs)))  # the total cost
        for i in range(n):
            self.add_propagator(([i, 2 * n - 2 + i], ALG_ELEMENT_IV, cost_rows[i]))
        self.add_propagator((list(range(2 * n - 2, 3 * n - 2)) + [3 * n - 2], ALG_AFFINE_EQ, [1] * n + [-1, 0]))
