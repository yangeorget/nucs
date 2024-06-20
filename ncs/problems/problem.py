from typing import Dict, List, Optional, Set, Tuple

import numba
import numpy as np
from numba import jit
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator

MIN = 0
MAX = 1


@jit(nopython=True)
def should_update(variables: NDArray, triggers: NDArray, changes: NDArray) -> bool:
    return changes is None or bool(np.any(changes[variables] & triggers))


@jit(nopython=True)
def compute_shr_changes(
    prop_indices: NDArray,
    prop_offsets: NDArray,
    prop_domains: NDArray,
    new_prop_domains: NDArray,
    shr_domains: NDArray,
) -> Tuple[Optional[NDArray], Optional[NDArray]]:
    new_shr_domains = np.zeros((len(shr_domains), 2), dtype=numba.int32)
    new_shr_domains[prop_indices, MIN] = np.maximum(new_prop_domains[:, MIN], prop_domains[:, MIN]) - prop_offsets
    new_shr_domains[prop_indices, MAX] = np.minimum(new_prop_domains[:, MAX], prop_domains[:, MAX]) - prop_offsets
    if np.any(np.greater(new_shr_domains[:, MIN], new_shr_domains[:, MAX])):
        return None, None
    return new_shr_domains, np.not_equal(new_shr_domains, shr_domains)


class Problem:
    """
    A problem is defined by a list of variable domains and a list of propagators.
    """

    def __init__(self, shr_domains: NDArray, dom_indices: List[int], dom_offsets: List[int]):
        self.shr_domains = shr_domains
        self.size = len(dom_indices)
        self.dom_indices = np.array(dom_indices)
        self.dom_offsets = np.array(dom_offsets)
        self.propagators: List[Propagator] = []

    def add_propagator(self, propagator: Propagator) -> None:
        propagator.offsets = self.dom_offsets[propagator.variables]
        propagator.indices = self.dom_indices[propagator.variables]
        self.propagators.append(propagator)

    def get_domains(self) -> NDArray:
        """
        Returns the domains of the problem variables.
        :return: an NDArray
        """
        return self.shr_domains[self.dom_indices] + self.dom_offsets.reshape(self.size, 1)

    def is_not_instantiated(self, var_idx: int) -> bool:
        """
        Returns a boolean indicating if a variable is not instantiated.
        :param var_idx: the index of the variable
        :return: True iff the variable is not instantiated
        """
        var_domain = self.dom_indices[var_idx]
        return bool(self.shr_domains[var_domain, MIN] < self.shr_domains[var_domain, MAX])

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
        :param changes: an array of boolean describing the variable domain changes
        :param statistics: where to record the statistics of the computation
        :return: False if the problem is not consistent
        """
        if statistics is not None:
            statistics["problem.filters.nb"] += 1
        propagators_to_filter: Set[Propagator] = set()
        self.update_propagators_to_filter(propagators_to_filter, changes, None)
        while len(propagators_to_filter) > 0:
            propagator = propagators_to_filter.pop()
            if statistics is not None:
                statistics["problem.propagators.filters.nb"] += 1
            new_changes = self.update_domains(propagator)
            if new_changes is None:
                return False
            self.update_propagators_to_filter(propagators_to_filter, new_changes, propagator)
        return True

    def update_domains(self, prop: Propagator) -> Optional[NDArray]:
        """
        Updates problem variable domains.
        :param prop: a propagator
        :return: a boolean array of variable changes
        """
        prop_domains = self.shr_domains[prop.indices] + prop.offsets.reshape(prop.size, 1)
        prop_ndomains = prop.compute_domains(prop_domains)
        if prop_ndomains is None:
            return None
        new_shr_domains, shr_changes = compute_shr_changes(prop.indices, prop.offsets, prop_domains, prop_ndomains, self.shr_domains)
        if shr_changes is None:
            return None
        self.shr_domains = new_shr_domains
        return shr_changes[self.dom_indices]

    def update_propagators_to_filter(
        self, propagators_to_filter: Set[Propagator], changes: Optional[NDArray], last_propagator: Optional[Propagator]
    ) -> None:
        """
        Updates the list of propagators that need to be filtered.
        :param propagators_to_filter: the list of propagators
        :param changes: an array of changes
        :param last_propagator: the last propagator that has been filtered
        """
        propagators_to_filter.update(
            propagator
            for propagator in self.propagators
            if (last_propagator is None or propagator != last_propagator)
            and should_update(propagator.variables, propagator.triggers, changes)
        )
