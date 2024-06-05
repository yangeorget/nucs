from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

MIN = 0
MAX = 1


class Problem:
    """
    A problem is defined by a set of variable domains and a set of propagators.
    """

    def __init__(self, shr_domains: NDArray, dom_indices: List[int], dom_offsets: List[int], propagators: List = []):
        self.shr_domains = shr_domains
        self.dom_indices = np.array(dom_indices)
        self.dom_offsets = np.array(dom_offsets)
        self.size = len(self.dom_indices)
        self.propagators = propagators

    def get_var_domains(self) -> NDArray:
        return self.shr_domains[self.dom_indices] + self.dom_offsets.reshape(self.size, 1)

    def get_lcl_domains(self, variables: NDArray) -> NDArray:
        # TODO: optimize
        return self.get_var_domains()[variables]

    def set_lcl_mins(self, variables: NDArray, lcl_mins: NDArray) -> None:
        self.shr_domains[self.dom_indices[variables], MIN] = lcl_mins - self.dom_offsets[variables]

    def set_lcl_maxs(self, variables: NDArray, lcl_maxs: NDArray) -> None:
        self.shr_domains[self.dom_indices[variables], MAX] = lcl_maxs - self.dom_offsets[variables]

    def is_not_instantiated(self, idx: int) -> bool:
        var_domain = self.dom_indices[idx]
        return self.shr_domains[var_domain, MIN] < self.shr_domains[var_domain, MAX]

    def is_inconsistent(self) -> bool:
        """
        Returns true iff the problem is consistent.
        :return: a boolean
        """
        return np.any(np.greater(self.shr_domains[:, MIN], self.shr_domains[:, MAX]))  # type: ignore

    def is_not_solved(self) -> bool:
        """
        Returns true iff the problem is not solved.
        :return: a boolean
        """
        return np.any(np.not_equal(self.shr_domains[:, MIN], self.shr_domains[:, MAX]))  # type: ignore

    def __str__(self) -> str:
        return f"domains={self.shr_domains}, propagators={self.propagators}"

    def filter(self, changes: Optional[NDArray] = None, statistics: Optional[Dict] = None) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached.
        :param changes: some initial domain changes
        :param statistics: where to record the statistics of the computation
        :return: false if the problem is not consistent
        """
        if changes is None:
            changes = np.ones((self.size, 2), dtype=bool)
        if statistics is not None:
            statistics["problem.filters.nb"] += 1
        while np.any(changes):
            new_changes = np.zeros((self.size, 2), dtype=bool)
            for propagator in self.propagators:
                if np.any(changes[propagator.variables] & propagator.triggers):
                    if not propagator.update_domains(self, new_changes):
                        return False
            changes = new_changes
        return True
