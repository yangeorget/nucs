from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numba import jit
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator

MIN = 0
MAX = 1


def update_propagators_to_filter(
    propagators_to_filter: Set[Propagator],
    propagators: List[Propagator],
    last_propagator: Optional[Propagator],
    shr_changes: Optional[NDArray],
) -> None:
    if shr_changes is None:
        propagators_to_filter.update(propagator for propagator in propagators if propagator != last_propagator)
    else:
        propagators_to_filter.update(
            propagator
            for propagator in propagators
            if propagator != last_propagator
            and should_be_filtered(propagator.triggers, propagator.indices, shr_changes)
        )


@jit(nopython=True, nogil=True)
def should_be_filtered(prop_triggers: NDArray, prop_indices: NDArray, shr_changes: NDArray) -> bool:
    """
    Return a boolean indicating if a propagator should be added to the set of propagators to be filtered.
    :param prop_triggers: the triggers of the propagator
    :param prop_indices: the shared domain indices of the propagator
    :param shr_changes: the shared domain changes
    :return: a boolean
    """
    return shr_changes is None or bool(np.any(shr_changes[prop_indices] & prop_triggers))


@jit(nopython=True, nogil=True)
def compute_propagator_domains(shr_domains: NDArray, prop_indices: NDArray, prop_offsets: NDArray) -> NDArray:
    """
    Computes the domains of the variables of a propagator.
    :param shr_domains: the shared domains
    :param prop_indices: the indices
    :param prop_offsets: the offsets
    :return: a NDArray of domains
    """
    prop_domains = shr_domains[prop_indices]
    prop_domains += prop_offsets.reshape(prop_indices.shape[0], 1)
    return prop_domains


@jit(nopython=True, nogil=True)
def compute_shared_domains_changes(
    prop_indices: NDArray,
    prop_offsets: NDArray,
    prop_domains: NDArray,
    new_prop_domains: NDArray,
    shr_domains: NDArray,
) -> Optional[NDArray]:
    """
    Computes the changes of the shared domains when a propagator is applied.
    :param prop_indices: the indices for the propagator variables
    :param prop_offsets: the offsets for the propagator variables
    :param prop_domains: the domains of the propagator variables
    :param new_prop_domains: the new domains of the propagator variables
    :param shr_domains: the shared domains
    :return: None if an inconsistency is detected or an NDArray of changes
    """
    if new_prop_domains is None:
        return None
    size = len(prop_indices)
    new_prop_mins = np.maximum(new_prop_domains[:, MIN], prop_domains[:, MIN])
    new_prop_maxs = np.minimum(new_prop_domains[:, MAX], prop_domains[:, MAX])
    if np.any(np.greater(new_prop_mins, new_prop_maxs)):
        return None
    old_shr_domains = shr_domains.copy()
    shr_domains[prop_indices] = np.hstack(
        (new_prop_mins.reshape(size, 1), new_prop_maxs.reshape(size, 1))
    ) - prop_offsets.reshape(size, 1)
    return np.not_equal(old_shr_domains, shr_domains)


class Problem:
    """
    A problem is defined by a list of variable domains and a list of propagators.
    """

    def __init__(self, shr_domains: List[Tuple[int, int]], dom_indices: List[int], dom_offsets: List[int]):
        self.size = len(dom_indices)
        self.shr_domains = np.array(shr_domains, dtype=np.int32).reshape(len(shr_domains), 2)
        self.dom_indices = np.array(dom_indices, dtype=np.uint16)
        self.dom_offsets = np.array(dom_offsets, dtype=np.int32)
        self.propagators: List[Propagator] = []
        self.propagators_to_filter: Set[Propagator] = set()

    def add_propagator(self, propagator: Propagator) -> None:
        propagator.offsets = self.dom_offsets[propagator.variables]
        propagator.indices = self.dom_indices[propagator.variables]
        self.propagators.append(propagator)

    def get_domains(self) -> NDArray:
        """
        Returns the domains of the problem variables.
        :return: an NDArray
        """
        domains = self.shr_domains[self.dom_indices]
        domains += self.dom_offsets.reshape(self.size, 1)
        return domains

    def get_values(self) -> List[int]:
        assert not self.is_not_solved()
        domains = self.get_domains()
        return domains[:, MIN].tolist()

    def is_not_instantiated(self, var_idx: int) -> bool:
        """
        Returns a boolean indicating if a variable is not instantiated.
        :param var_idx: the index of the variable
        :return: True iff the variable is not instantiated
        """
        domain = self.shr_domains[self.dom_indices[var_idx]]
        return bool(domain[MIN] < domain[MAX])

    def is_not_solved(self) -> bool:
        """
        Returns true iff the problem is not solved.
        :return: a boolean
        """
        return bool(np.any(np.not_equal(self.shr_domains[:, MIN], self.shr_domains[:, MAX])))

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
        self.propagators_to_filter.clear()
        update_propagators_to_filter(self.propagators_to_filter, self.propagators, None, changes)
        while bool(self.propagators_to_filter):
            propagator = self.propagators_to_filter.pop()
            if statistics is not None:
                statistics["problem.propagators.filters.nb"] += 1
            prop_domains = compute_propagator_domains(self.shr_domains, propagator.indices, propagator.offsets)
            new_prop_domains = propagator.compute_domains(prop_domains)
            shr_changes = compute_shared_domains_changes(
                propagator.indices, propagator.offsets, prop_domains, new_prop_domains, self.shr_domains
            )
            if shr_changes is None:
                return False
            update_propagators_to_filter(self.propagators_to_filter, self.propagators, propagator, shr_changes)
        return True

    def pretty_print(self, solution: List[int]) -> None:
        print(solution)
