from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN, Problem


class Propagator:
    """
    Abstraction for a bound-consistency algorithm.
    """

    def __init__(self, variables: List[int]):
        self.size = len(variables)
        self.variables = np.array(variables)
        self.triggers = np.ones((self.size, 2), dtype=bool)

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        """
        Computes new domains for the variables
        :param domains: the initial domains
        :return: the new domains or None if there is an inconsistency
        """
        pass

    def update_domains(self, problem: Problem, changes: NDArray) -> bool:
        """
        Updates problem variable domains.
        :param problem: a problem
        :param changes: where to record the domain changes
        :return: false if the problem is not consistent
        """
        lcl_domains = problem.get_lcl_domains(self.variables)
        new_domains = self.compute_domains(lcl_domains)
        if new_domains is None:
            return False
        lcl_changes = np.full((len(new_domains), 2), False)
        lcl_mins = lcl_domains[:, MIN]
        new_minimums = np.maximum(new_domains[:, MIN], lcl_mins)
        np.greater(new_minimums, lcl_mins, out=lcl_changes[:, MIN])
        problem.set_lcl_mins(self.variables, new_minimums)
        if problem.is_inconsistent():
            return False
        lcl_maxs = lcl_domains[:, MAX]
        new_maximums = np.minimum(new_domains[:, MAX], lcl_maxs)
        np.less(new_maximums, lcl_maxs, out=lcl_changes[:, MAX])
        problem.set_lcl_maxs(self.variables, new_maximums)
        changes[self.variables] |= lcl_changes
        if problem.is_inconsistent():
            return False
        return True
