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
from typing import List

from nucs.problems.circuit_problem import CircuitProblem
from nucs.propagators.propagators import ALG_ELEMENT_EQ, ALG_SUM_EQ


class TSPProblem(CircuitProblem):
    """ """

    def __init__(self, costs: List[List[int]]) -> None:
        """
        Initializes the problem.
        :param costs: the costs between vertices as a list of lists of integers
        """
        n = len(costs)
        super().__init__(n)
        max_costs = [max(cost_row) for cost_row in costs]
        min_costs = [min([cost for cost in cost_row if cost > 0]) for cost_row in costs]
        self.succ_costs = self.add_variables([(min_costs[i], max_costs[i]) for i in range(n)])
        self.pred_costs = self.add_variables([(min_costs[i], max_costs[i]) for i in range(n)])
        self.total_cost = self.add_variable((sum(min_costs), sum(max_costs)))  # the total cost
        for i in range(n):
            self.add_propagator(ALG_ELEMENT_EQ, [i, self.succ_costs + i], costs[i])
            self.add_propagator(ALG_ELEMENT_EQ, [n + i, self.pred_costs + i], costs[i])
        self.add_propagator(ALG_SUM_EQ, list(range(self.succ_costs, self.succ_costs + n)) + [self.total_cost])
        self.add_propagator(ALG_SUM_EQ, list(range(self.pred_costs, self.pred_costs + n)) + [self.total_cost])
        # total_cost_prop_idx = register_propagator(
        #     get_triggers_total_cost, get_complexity_total_cost, compute_domains_total_cost
        # )
        # costs = [cost for cost_row in costs for cost in cost_row]
        # self.add_propagator((list(range(n)) + [self.total_cost], total_cost_prop_idx, costs))
        # self.add_propagator((list(range(n, 2 * n)) + [self.total_cost], total_cost_prop_idx, costs))
