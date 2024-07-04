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
        self.domain_indices = np.array(domain_indices, dtype=np.int16)
        self.domain_offsets = np.array(domain_offsets, dtype=np.int32)

    def set_propagators(self, propagators: List[Tuple[List[int], int]]) -> None:
        self.propagator_nb = len(propagators)
        self.propagators_to_filter = np.zeros(self.propagator_nb, dtype=np.bool)
        self.propagator_algorithms = np.empty(self.propagator_nb, dtype=np.int8)
        propagator_variables = []
        self.propagator_sizes = np.empty(self.propagator_nb, dtype=np.int16)
        self.propagator_starts = np.empty(self.propagator_nb, dtype=np.int16)
        self.propagator_ends = np.empty(self.propagator_nb, dtype=np.int16)
        self.propagator_starts[0] = 0
        for propagator_idx, propagator in enumerate(propagators):
            prop_variables = propagator[0]
            prop_algorithm = propagator[1]
            propagator_size = len(prop_variables)
            propagator_variables.append(prop_variables)
            self.propagator_sizes[propagator_idx] = propagator_size
            self.propagator_algorithms[propagator_idx] = prop_algorithm
            if propagator_idx > 0:
                self.propagator_starts[propagator_idx] = self.propagator_ends[propagator_idx - 1]
            self.propagator_ends[propagator_idx] = self.propagator_starts[propagator_idx] + propagator_size
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
        assert not is_not_solved(self.shared_domains)
        domains = self.get_domains()
        return domains[:, MIN].tolist()


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
            self.propagators_to_filter,
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


@jit(nopython=True, nogil=True, cache=True)
def filter(
    propagator_nb: int,
    propagators_to_filter: NDArray,
    propagator_algorithms: NDArray,
    propagator_starts: NDArray,
    propagator_ends: NDArray,
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
    init_propagators_to_filter(
        propagators_to_filter,
        changes,
        propagator_nb,
        propagator_starts,
        propagator_ends,
        propagator_indices,
        propagator_triggers,
    )
    while True:
        found = False
        for prop_idx in range(propagator_nb):
            if propagators_to_filter[prop_idx]:
                propagators_to_filter[prop_idx] = False
                found = True
                break
        if not found:
            return True
        statistics[STATS_PROBLEM_PROPAGATORS_FILTERS_NB] += 1
        prop_start = propagator_starts[prop_idx]
        prop_end = propagator_ends[prop_idx]
        prop_indices = propagator_indices[prop_start:prop_end]
        prop_offsets = propagator_offsets[prop_start:prop_end].reshape((-1, 1))
        prop_new_domains = compute_domains(propagator_algorithms[prop_idx], shared_domains[prop_indices] + prop_offsets)
        if prop_new_domains is None:
            return False
        old_shared_domains = shared_domains.copy()
        shared_domains[prop_indices] = prop_new_domains - prop_offsets
        update_propagators_to_filter(
            propagators_to_filter,
            np.not_equal(old_shared_domains, shared_domains),
            propagator_nb,
            propagator_starts,
            propagator_ends,
            propagator_indices,
            propagator_triggers,
            prop_idx,
        )


@jit(nopython=True, nogil=True, cache=True)
def init_propagators_to_filter(
    propagators_to_filter,
    changes,
    propagator_nb,
    propagator_starts,
    propagator_ends,
    propagator_indices,
    propagator_triggers,
):
    if changes is None:  # this is an initialization
        propagators_to_filter.fill(True)
    else:
        for prop_idx in range(propagator_nb):
            prop_start = propagator_starts[prop_idx]
            prop_end = propagator_ends[prop_idx]
            propagators_to_filter[prop_idx] = np.any(
                changes[propagator_indices[prop_start:prop_end]] & propagator_triggers[prop_start:prop_end]
            )


@jit(nopython=True, nogil=True, cache=True)
def update_propagators_to_filter(
    propagators_to_filter,
    changes,
    propagator_nb,
    propagator_starts,
    propagator_ends,
    propagator_indices,
    propagator_triggers,
    propagator_idx,
):
    for prop_idx in range(propagator_nb):
        if prop_idx != propagator_idx:
            prop_start = propagator_starts[prop_idx]
            prop_end = propagator_ends[prop_idx]
            if np.any(changes[propagator_indices[prop_start:prop_end]] & propagator_triggers[prop_start:prop_end]):
                propagators_to_filter[prop_idx] = True


@jit(nopython=True, nogil=True, cache=True)
def compute_domains(algorithm: int, propagator_domains: NDArray) -> Optional[NDArray]:
    if algorithm == ALGORITHM_ALLDIFFERENT_LOPEZ_ORTIZ:
        return alldifferent_lopez_ortiz_propagator.compute_domains(propagator_domains)
    elif algorithm == ALGORITHM_SUM:
        return sum_propagator.compute_domains(propagator_domains)
    elif algorithm == ALGORITHM_DUMMY:
        return dummy_propagator.compute_domains(propagator_domains)
    return None

@jit(nopython=True, nogil=True, cache=True)
def is_not_solved(shared_domains: NDArray) -> bool:
    """
    Returns true iff the problem is not solved.
    :return: a boolean
    """
    return bool(np.any(np.not_equal(shared_domains[:, MIN], shared_domains[:, MAX])))

@jit(nopython=True, nogil=True, cache=True)
def not_instantiated_index(variable_nb: int, shared_domains: NDArray, domain_indices: NDArray) -> int:
    for var_idx in range(variable_nb):
        domain = shared_domains[domain_indices[var_idx]]
        if domain[MIN] < domain[MAX]:
            return var_idx
    return -1

