from typing import Optional, Tuple

import numpy as np
from numba import jit  # type: ignore
from numba.typed import List
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator
from ncs.utils import (
    MAX,
    MIN,
    STATS_PROBLEM_FILTERS_NB,
    STATS_PROBLEM_PROPAGATORS_FILTERS_NB,
    stats_init,
)


def filter(
    propagators: List[Propagator], shr_domains: NDArray, statistics: NDArray, changes: Optional[NDArray]
) -> bool:
    """
    Filters the problem's domains by applying the propagators until a fix point is reached.
    :param statistics: where to record the statistics of the computation
    :return: False if the problem is not consistent
    """
    statistics[STATS_PROBLEM_FILTERS_NB] += 1
    if changes is None:  # this is an initialization
        propagators_to_filter = np.ones(len(propagators), dtype=np.bool)  # TODO: create once
    else:
        propagators_to_filter = np.zeros(len(propagators), dtype=np.bool)
        update_propagators_to_filter(propagators_to_filter, propagators, -1, changes)
    while (propagator_idx := pop_propagator_to_filter(propagators_to_filter)) != -1:
        propagator = propagators[propagator_idx]
        statistics[STATS_PROBLEM_PROPAGATORS_FILTERS_NB] += 1
        prop_domains = compute_propagator_domains(shr_domains, propagator.indices, propagator.offsets)  # TODO: include in propagator
        new_prop_domains = propagator.compute_domains(prop_domains)
        shr_changes = compute_shared_domains_changes(
            propagator.indices, propagator.offsets, prop_domains, new_prop_domains, shr_domains  # TODO: include in propagator
        )
        if shr_changes is None:
            return False
        update_propagators_to_filter(propagators_to_filter, propagators, propagator_idx, shr_changes)
    return True


def update_propagators_to_filter(
    propagators_to_filter: NDArray,
    propagators: List[Propagator],
    last_propagator_idx: int,
    shr_changes: NDArray,
) -> None:
    for propagator_idx, propagator in enumerate(propagators):
        if propagator_idx != last_propagator_idx and should_be_filtered(
            propagator.triggers, propagator.indices, shr_changes
        ):
            propagators_to_filter[propagator_idx] = True


@jit(nopython=True, nogil=True, cache=True)
def pop_propagator_to_filter(propagators_to_filter: NDArray) -> int:
    if np.any(propagators_to_filter):
        propagator_idx = int(np.argmax(propagators_to_filter))
        propagators_to_filter[propagator_idx] = False
        return propagator_idx
    else:
        return -1


@jit(nopython=True, nogil=True, cache=True)
def should_be_filtered(prop_triggers: NDArray, prop_indices: NDArray, shr_changes: NDArray) -> bool:
    """
    Return a boolean indicating if a propagator should be added to the set of propagators to be filtered.
    :param prop_triggers: the triggers of the propagator
    :param prop_indices: the shared domain indices of the propagator
    :param shr_changes: the shared domain changes
    :return: a boolean
    """
    return shr_changes is None or bool(np.any(shr_changes[prop_indices] & prop_triggers))


@jit(nopython=True, nogil=True, cache=True)
def compute_propagator_domains(shr_domains: NDArray, prop_indices: NDArray, prop_offsets: NDArray) -> NDArray:
    """
    Computes the domains of the variables of a propagator.
    :param shr_domains: the shared domains
    :param prop_indices: the indices
    :param prop_offsets: the offsets
    :return: a NDArray of domains
    """
    prop_domains = shr_domains[prop_indices]
    prop_domains += prop_offsets.reshape((-1, 1))
    return prop_domains


@jit(nopython=True, nogil=True, cache=True)
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
    new_prop_mins = np.maximum(new_prop_domains[:, MIN], prop_domains[:, MIN])
    new_prop_maxs = np.minimum(new_prop_domains[:, MAX], prop_domains[:, MAX])
    if np.any(np.greater(new_prop_mins, new_prop_maxs)):
        return None
    old_shr_domains = shr_domains.copy()
    shr_domains[prop_indices] = np.hstack(
        (new_prop_mins.reshape((-1, 1)), new_prop_maxs.reshape((-1, 1)))
    ) - prop_offsets.reshape((-1, 1))
    return np.not_equal(old_shr_domains, shr_domains)


class Problem:
    """
    A problem is defined by a list of variable domains and a list of propagators.
    """

    def __init__(self, shr_domains: List[Tuple[int, int]], dom_indices: List[int], dom_offsets: List[int]):
        self.size = len(dom_indices)
        self.shr_domains = np.array(shr_domains, dtype=np.int32).reshape((-1, 2))
        self.dom_indices = np.array(dom_indices, dtype=np.int32)
        self.dom_offsets = np.array(dom_offsets, dtype=np.int32)
        self.propagators: List[Propagator] = []

    def add_propagator(self, propagator: Propagator) -> None:
        propagator.offsets = self.dom_offsets[propagator.variables].reshape(-1)
        propagator.indices = self.dom_indices[propagator.variables].reshape(-1)
        self.propagators.append(propagator)

    def get_domains(self) -> NDArray:
        """
        Returns the domains of the problem variables.
        :return: an NDArray
        """
        domains = self.shr_domains[self.dom_indices]
        domains += self.dom_offsets.reshape((-1, 1))
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

    def filter(self, statistics: NDArray = stats_init(), changes: Optional[NDArray] = None) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached.
        :param statistics: where to record the statistics of the computation
        :return: False if the problem is not consistent
        """
        return filter(self.propagators, self.shr_domains, statistics, changes)

    def pretty_print(self, solution: List[int]) -> None:
        print(solution)
