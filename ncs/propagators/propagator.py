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
        domains = problem.get_domains()  # TODO: optimize ?
        var_domains = domains[self.variables]
        new_domains = self.compute_domains(var_domains)
        if new_domains is None:
            return False
        var_changes = np.full((len(new_domains), 2), False)
        var_offsets = problem.dom_offsets[self.variables]  # TODO: could be computed at init time
        var_indices = problem.dom_indices[self.variables]  # TODO: could be computed at init time
        np.greater(new_domains[:, MIN], var_domains[:, MIN], out=var_changes[:, MIN])
        problem.shr_domains[var_indices, MIN] = np.maximum(new_domains[:, MIN], var_domains[:, MIN]) - var_offsets
        if problem.is_inconsistent():
            return False
        np.less(new_domains[:, MAX], var_domains[:, MAX], out=var_changes[:, MAX])
        problem.shr_domains[var_indices, MAX] = np.minimum(new_domains[:, MAX], var_domains[:, MAX]) - var_offsets
        if problem.is_inconsistent():
            return False
        changes |= var_changes[problem.dom_indices]
        return True
