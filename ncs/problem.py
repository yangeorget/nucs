from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator

MIN = 0
MAX = 1


class Problem:
    """
    A problem is defined by a set of variable domains and a set of propagators.
    """

    def __init__(self, domains: NDArray):
        self.domains = domains
        self.propagators: List = []

    def filter(self, changes: Optional[NDArray], statistics: Optional[Dict] = None) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached.
        :param changes: some initial domain changes
        :param statistics: where to record the statistics of the computation
        :return: false if the problem is not consistent
        """
        # print("filter()")
        if changes is None:
            changes = np.ones((len(self.domains), 2), dtype=bool)
        if statistics is not None:
            statistics["problem.filters.nb"] += 1
        while np.any(changes):
            propagators = self.propagators  # TODO get from changes
            changes = np.zeros((len(self.domains), 2), dtype=bool)
            for propagator in propagators:
                self.update_domains(propagator, changes)
                if self.is_inconsistent():
                    return False
        return True

    def update_domains(self, propagator: Propagator, changes: NDArray) -> None:
        """
        Updates the problem's domains.
        :param propagator: a propagato
        :param changes: where to record the domain changes
        :return: the nx2 matrix of changes
        """
        # print(f"update_domains({variables}, {new_domains})")
        new_domains = propagator.compute_domains(self.domains)
        local_changes = np.full((len(new_domains), 2), False)
        new_minimums = np.maximum(new_domains[:, MIN], self.domains[propagator.variables, MIN])
        np.greater(new_minimums, self.domains[propagator.variables, MIN], out=local_changes[:, MIN])
        self.domains[propagator.variables, MIN] = new_minimums
        new_maximums = np.minimum(new_domains[:, MAX], self.domains[propagator.variables, MAX])
        np.less(new_maximums, self.domains[propagator.variables, MAX], out=local_changes[:, MAX])
        self.domains[propagator.variables, MAX] = new_maximums
        changes[propagator.variables] |= local_changes

    def is_inconsistent(self) -> bool:
        """
        Returns true iff the problem is consistent.
        :return: a boolean
        """
        # print("is_inconsistent()")
        return np.any(np.greater(self.domains[:, MIN], self.domains[:, MAX]))  # type: ignore

    def is_not_solved(self) -> bool:
        """
        Returns true iff the problem is not solved.
        :return: a boolean
        """
        # print("is_solved()")
        return np.any(np.not_equal(self.domains[:, MIN], self.domains[:, MAX]))  # type: ignore

    def __str__(self) -> str:
        return f"domains={self.domains}, propagators={self.propagators}"
