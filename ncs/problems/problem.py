from typing import Optional, Tuple

import numpy as np
from numba import jit  # type: ignore
from numba.typed import List
from numpy.typing import NDArray

from ncs.propagators import (
    alldifferent_lopez_ortiz_propagator,
    dummy_propagator,
    sum_propagator,
)
from ncs.utils import (
    MAX,
    MIN,
    STATS_PROBLEM_FILTERS_NB,
    STATS_PROBLEM_PROPAGATORS_FILTERS_NB,
    statistics_init,
)

ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ = 0
ALGORITHM_DUMMY = 1
ALGORITHM_SUM = 2


class Problem:
    """
    A problem is defined by a list of variable domains and a list of propagators.
    """

    def __init__(self, shared_domains: List[Tuple[int, int]], domain_indices: List[int], domain_offsets: List[int]):
        self.variable_nb = len(domain_indices)
        self.shared_domains = np.array(shared_domains, dtype=np.int32).reshape((-1, 2))
        self.domain_indices = np.array(domain_indices, dtype=np.int32)
        self.domain_offsets = np.array(domain_offsets, dtype=np.int32)

    def set_propagators(self, propagators: List[Tuple[List[int], int]]) -> None:
        self.propagator_nb = len(propagators)
        self.propagator_algorithms: List[int] = []
        propagator_variables = []
        self.propagator_sizes: List[int] = []
        self.propagator_starts: List[int] = []
        self.propagator_ends: List[int] = []
        self.propagator_starts.append(0)
        for propagator_idx, propagator in enumerate(propagators):
            prop_variables = propagator[0]
            prop_algorithm = propagator[1]
            propagator_size = len(prop_variables)
            propagator_variables.append(prop_variables)
            self.propagator_sizes.append(propagator_size)
            self.propagator_algorithms.append(prop_algorithm)
            if propagator_idx > 0:
                self.propagator_starts.append(self.propagator_ends[propagator_idx - 1])
            self.propagator_ends.append(self.propagator_starts[propagator_idx] + propagator_size)
        self.propagator_total_size = sum(self.propagator_sizes)
        self.propagator_variables = np.array(propagator_variables, dtype=np.int16)
        self.propagator_indices = np.empty(self.propagator_total_size, dtype=np.int16)
        self.propagator_offsets = np.empty(self.propagator_total_size, dtype=np.int32)
        for propagator_idx, propagator in enumerate(propagators):
            prop_variables = propagator[0]
            self.propagator_indices[self.propagator_starts[propagator_idx] : self.propagator_ends[propagator_idx]] = (
                self.domain_indices[prop_variables]
            )
            self.propagator_offsets[self.propagator_starts[propagator_idx] : self.propagator_ends[propagator_idx]] = (
                self.domain_offsets[prop_variables]
            )
        self.propagator_triggers = np.ones((self.propagator_total_size, 2), dtype=np.bool)

    def get_domains(self) -> NDArray:
        """
        Returns the domains of the problem variables.
        :return: an NDArray
        """
        domains = self.shared_domains[self.domain_indices]
        domains += self.domain_offsets.reshape((-1, 1))
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
        domain = self.shared_domains[self.domain_indices[var_idx]]
        return bool(domain[MIN] < domain[MAX])

    def is_not_solved(self) -> bool:
        """
        Returns true iff the problem is not solved.
        :return: a boolean
        """
        return bool(np.any(np.not_equal(self.shared_domains[:, MIN], self.shared_domains[:, MAX])))

    def __str__(self) -> str:
        return f"domains={self.shared_domains}"

    def filter(self, statistics: NDArray = statistics_init(), changes: Optional[NDArray] = None) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached.
        :param statistics: where to record the statistics of the computation
        :return: False if the problem is not consistent
        """
        return filter(
            self.propagator_nb,
            self.propagator_algorithms,
            self.propagator_starts,
            self.propagator_ends,
            self.propagator_indices,
            self.propagator_offsets,
            self.propagator_triggers,
            self.shared_domains,
            statistics,
            changes,
        )

    def pretty_print(self, solution: List[int]) -> None:
        print(solution)

#@jit(nopython=True, nogil=True, cache=True)
def filter(
    propagator_nb: int,
    propagator_algorithms: List[int],
    propagator_starts: List[int],
    propagator_ends: List[int],
    propagator_indices: NDArray,
    propagator_offsets: NDArray,
    propagator_triggers: NDArray,
    shared_domains: NDArray,
    statistics: NDArray,
    changes: Optional[NDArray],
) -> bool:
    """
    Filters the problem's domains by applying the propagators until a fix point is reached.
    :param statistics: where to record the statistics of the computation
    :return: False if the problem is not consistent
    """
    statistics[STATS_PROBLEM_FILTERS_NB] += 1
    if changes is None:  # this is an initialization
        propagators_to_filter = np.ones(propagator_nb, dtype=np.bool)  # TODO: create once
    else:
        propagators_to_filter = np.zeros(propagator_nb, dtype=np.bool)
        update_propagators_to_filter(
            propagators_to_filter, propagator_nb, propagator_triggers, propagator_indices, -1, changes
        )
    while np.any(propagators_to_filter):
        propagator_idx = int(np.argmax(propagators_to_filter))
        propagators_to_filter[propagator_idx] = False
        statistics[STATS_PROBLEM_PROPAGATORS_FILTERS_NB] += 1
        propagator_domains = shared_domains[
            propagator_indices[propagator_starts[propagator_idx] : propagator_ends[propagator_idx]]
        ] + propagator_offsets[propagator_starts[propagator_idx] : propagator_ends[propagator_idx]].reshape((-1, 1))
        new_propagator_domains = compute_domains(propagator_algorithms[propagator_idx], propagator_domains)
        shared_changes = compute_shared_domains_changes(
            propagator_indices[propagator_starts[propagator_idx] : propagator_ends[propagator_idx]],
            propagator_offsets[propagator_starts[propagator_idx] : propagator_ends[propagator_idx]],
            propagator_domains,
            new_propagator_domains,
            shared_domains,
        )
        if shared_changes is None:
            return False
        update_propagators_to_filter(
            propagators_to_filter,
            propagator_nb,
            propagator_triggers,
            propagator_indices,
            propagator_idx,
            shared_changes,
        )
    return True

#@jit(nopython=True, nogil=True, cache=True)
def update_propagators_to_filter(
    propagators_to_filter: NDArray,
    propagator_nb: int,
    propagator_triggers: NDArray,
    propagator_indices: NDArray,
    last_propagator_idx: int,
    shared_changes: NDArray,
) -> None:
    for propagator_idx in range(0, propagator_nb):
        if (
            propagator_idx != last_propagator_idx
            and (shared_changes is None or bool(np.any(shared_changes[propagator_indices] & propagator_triggers)))
        ):
            propagators_to_filter[propagator_idx] = True


#@jit(nopython=True, nogil=True, cache=True)
def compute_domains(algorithm: int, domains: NDArray) -> Optional[NDArray]:
    if algorithm == ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ:
        return alldifferent_lopez_ortiz_propagator.compute_domains(domains)
    if algorithm == ALGORITHM_SUM:
        return sum_propagator.compute_domains(domains)
    if algorithm == ALGORITHM_DUMMY:
        return dummy_propagator.compute_domains(domains)
    return None


#@jit(nopython=True, nogil=True, cache=True)
def compute_shared_domains_changes(
    propagator_indices: NDArray,
    propagator_offsets: NDArray,
    propagator_domains: NDArray,
    new_propagator_domains: NDArray,
    shared_domains: NDArray,
) -> Optional[NDArray]:
    """
    Computes the changes of the shared domains when a propagator is applied.
    :param propagator_indices: the indices for the propagator variables
    :param propagator_offsets: the offsets for the propagator variables
    :param propagator_domains: the domains of the propagator variables
    :param new_propagator_domains: the new domains of the propagator variables
    :param shared_domains: the shared domains
    :return: None if an inconsistency is detected or an NDArray of changes
    """
    if new_propagator_domains is None:
        return None
    new_prop_mins = np.maximum(new_propagator_domains[:, MIN], propagator_domains[:, MIN])
    new_prop_maxs = np.minimum(new_propagator_domains[:, MAX], propagator_domains[:, MAX])
    if np.any(np.greater(new_prop_mins, new_prop_maxs)):
        return None
    old_shr_domains = shared_domains.copy()
    shared_domains[propagator_indices] = np.hstack(
        (new_prop_mins.reshape((-1, 1)), new_prop_maxs.reshape((-1, 1)))
    ) - propagator_offsets.reshape((-1, 1))
    return np.not_equal(old_shr_domains, shared_domains)
