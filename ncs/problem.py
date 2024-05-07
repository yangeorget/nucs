from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

MIN = 0
MAX = 1


class Problem:
    def __init__(self, domains: NDArray):
        self.domains = domains
        self.propagators: List = []

    def filter(self, statistics: Optional[Dict] = None) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached
        :return: false if the problem is not consistent
        """
        # print("filter()")
        if statistics is not None:
            statistics["problem.filters.nb"] += 1
        changes = True
        while changes:
            changes = False
            for propagator in self.propagators:
                new_domains = propagator.compute_domains(self.domains)
                propagator_changes = self.update_domains(propagator.variables, new_domains)
                # TODO: take into account changes to filter a subset of all propagators
                if self.is_inconsistent():
                    return False
                changes |= np.any(propagator_changes)  # type: ignore
        return True

    def update_domains(self, variables: NDArray, new_domains: NDArray) -> NDArray:
        """
        Updates the problem's domains
        :param variables: some variables
        :param new_domains: the corresponding new domains
        :return: the nx2 matrix of changes
        """
        # print(f"update_domains({variables}, {new_domains})")
        changes = np.full((len(new_domains), 2), False)
        new_minimums = np.maximum(new_domains[:, MIN], self.domains[variables, MIN])
        np.greater(new_minimums, self.domains[variables, MIN], out=changes[:, MIN])
        self.domains[variables, MIN] = new_minimums
        new_maximums = np.minimum(new_domains[:, MAX], self.domains[variables, MAX])
        np.less(new_maximums, self.domains[variables, MAX], out=changes[:, MAX])
        self.domains[variables, MAX] = new_maximums
        return changes

    def is_instantiated(self, idx: int) -> bool:
        """
        Returns true iff the ith variable is instantiated
        :param idx: the index of the variable
        :return: a boolean
        """
        # print("is_instantiated()")
        return self.domains[idx, MIN] == self.domains[idx, MAX]

    def is_inconsistent(self) -> bool:
        """
        Returns true iff the problem is consistent
        :return: a boolean
        """
        # print("is_inconsistent()")
        return np.any(np.greater(self.domains[:, MIN], self.domains[:, MAX]))  # type: ignore

    def is_solved(self) -> bool:
        """
        Returns true iff the problem is solved
        :return: a boolean
        """
        # print("is_solved()")
        return np.all(np.equal(self.domains[:, MIN], self.domains[:, MAX]))  # type: ignore

    def __str__(self) -> str:
        return f"domains={self.domains}, propagators={self.propagators}"
