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

from nucs.examples.tsp.total_cost_propagator import (
    compute_domains_total_cost,
    get_complexity_total_cost,
    get_triggers_total_cost,
)
from nucs.problems.circuit_problem import CircuitProblem
from nucs.propagators.propagators import ALG_AFFINE_EQ, ALG_ELEMENT_IV, register_propagator


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
        next_costs = self.add_variables([(min_costs[i], max_costs[i]) for i in range(n)])
        prev_costs = self.add_variables([(min_costs[i], max_costs[i]) for i in range(n)])
        total_cost = self.add_variable((sum(min_costs), sum(max_costs)))  # the total cost
        for i in range(n):
            self.add_propagator(([i, next_costs + i], ALG_ELEMENT_IV, cost_rows[i]))
            self.add_propagator(([n + i, prev_costs + i], ALG_ELEMENT_IV, cost_rows[i]))
        self.add_propagator((list(range(next_costs, next_costs + n)) + [total_cost], ALG_AFFINE_EQ, [1] * n + [-1, 0]))
        self.add_propagator((list(range(prev_costs, prev_costs + n)) + [total_cost], ALG_AFFINE_EQ, [1] * n + [-1, 0]))
        total_cost_prop_idx = register_propagator(
            get_triggers_total_cost, get_complexity_total_cost, compute_domains_total_cost
        )
        costs = [cost for cost_row in cost_rows for cost in cost_row]
        self.add_propagator((list(range(n)) + [total_cost], total_cost_prop_idx, costs))
        self.add_propagator((list(range(n, 2 * n)) + [total_cost], total_cost_prop_idx, costs))
