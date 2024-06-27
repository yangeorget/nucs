import numpy as np

from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problems.problem import Problem
from ncs.utils import MAX, MIN


class SmallestDomainVariableHeuristic(VariableHeuristic):
    """
    Chooses the variable with the smallest domain which is not instantiated.
    """

    def choose_variable(self, problem: Problem) -> int:
        valid_domain_sizes = np.ma.masked_equal(
            problem.shr_domains[problem.dom_indices, MAX] - problem.shr_domains[problem.dom_indices, MIN], 0, copy=False
        )
        if valid_domain_sizes.count() == 0:
            return -1
        return np.ma.argmin(valid_domain_sizes)
