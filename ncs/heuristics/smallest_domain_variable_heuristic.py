import numpy as np

from ncs.heuristics.variable_heuristic import VariableHeuristic
from ncs.problems.problem import MAX, MIN, Problem


class SmallestDomainVariableHeuristic(VariableHeuristic):
    """
    Chooses the variable with the smallest domain which is not instantiated.
    """

    def choose_variable(self, problem: Problem) -> int:
        non_zero_domain_sizes = np.ma.masked_equal(problem.domains[:, MAX] - problem.domains[:, MIN], 0, copy=False)
        if non_zero_domain_sizes.count() == 0:
            return -1
        return np.ma.argmin(non_zero_domain_sizes)
