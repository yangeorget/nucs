from typing import Dict, List, Optional, Set, Tuple

import numba
import numpy as np
from numba import jit
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator

MIN = 0
MAX = 1


@jit(nopython=True, nogil=True)
def should_be_filtered(triggers: NDArray, indices: NDArray, shr_changes: NDArray) -> bool:
    """
    Return a boolean indicating if a propagator should be added to the set of propagators to be filtered.
    :param triggers: the triggers of the propagator
    :param indices: the shared domain indices of the propagator
    :param shr_changes: the shared domain changes
    :return: a boolean
    """
    return shr_changes is None or bool(np.any(shr_changes[indices] & triggers))


@jit(nopython=True, nogil=True)
def compute_prop_domains(shr_domains: NDArray, prop_indices: NDArray, prop_offsets: NDArray) -> NDArray:
    prop_domains = shr_domains[prop_indices]
    prop_domains += prop_offsets.reshape(prop_indices.shape[0], 1)
    return prop_domains


@jit(nopython=True, nogil=True)
def compute_shr_changes(
    prop_indices: NDArray,
    prop_offsets: NDArray,
    prop_domains: NDArray,
    new_prop_domains: NDArray,
    shr_domains: NDArray,
) -> Tuple[Optional[NDArray], Optional[NDArray]]:
    if new_prop_domains is None:
        return None, None
    new_prop_bounds = np.empty((len(prop_domains),2), dtype=numba.int32)
    new_prop_bounds[:, MIN] = np.maximum(new_prop_domains[:, MIN], prop_domains[:, MIN])
    new_prop_bounds[:, MAX] = np.minimum(new_prop_domains[:, MAX], prop_domains[:, MAX])
    if np.any(np.greater(new_prop_bounds[:, MIN], new_prop_bounds[:, MAX])):
        return None, None
    new_prop_bounds[:, MIN] -= prop_offsets
    new_prop_bounds[:, MAX] -= prop_offsets
    new_shr_domains = shr_domains.copy()
    new_shr_domains[prop_indices] = new_prop_bounds
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
            shr_changes = self.update_domains(propagator)
            if shr_changes is None:
                return False
            self.update_propagators_to_filter(propagators_to_filter, shr_changes, propagator)
        return True

    def update_domains(self, prop: Propagator) -> Optional[NDArray]:
        """
        Updates problem variable domains.
        :param prop: a propagator
        :return: a boolean array of variable changes
        """
        prop_domains = compute_prop_domains(self.shr_domains, prop.indices, prop.offsets)
        new_prop_domains = prop.compute_domains(prop_domains)
        new_shr_domains, shr_changes = compute_shr_changes(
            prop.indices, prop.offsets, prop_domains, new_prop_domains, self.shr_domains
        )
        self.shr_domains = new_shr_domains
        return shr_changes

    def update_propagators_to_filter(
        self,
        propagators_to_filter: Set[Propagator],
        shr_changes: Optional[NDArray],
        last_propagator: Optional[Propagator],
    ) -> None:
        """
        Updates the list of propagators that need to be filtered.
        :param propagators_to_filter: the list of propagators
        :param shr_changes: an array of changes
        :param last_propagator: the last propagator that has been filtered
        """
        propagators_to_filter.update(
            propagator
            for propagator in self.propagators
            if (last_propagator is None or propagator != last_propagator)
            and should_be_filtered(propagator.triggers, propagator.indices, shr_changes)
        )
