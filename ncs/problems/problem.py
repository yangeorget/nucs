from typing import Optional, Tuple, Union

import numpy as np
from numba import jit  # type: ignore
from numba.typed import List
from numpy.typing import NDArray

from ncs.propagators.propagators import (
    compute_domains,
    get_triggers,
    init_propagators_to_filter,
    update_propagators_to_filter,
)
from ncs.utils import (
    END,
    MAX,
    MIN,
    START,
    STATS_PROBLEM_FILTER_NB,
    STATS_PROPAGATOR_FILTER_NB,
    stats_init,
)


class Problem:
    """
    A problem is conceptually defined by a list of domains, a list of variables and a list of propagators.
    A domain can be computed by applying an offset to a shared domain.
    """

    def __init__(self, shr_domains: List[Union[int, Tuple[int, int]]], dom_indices: List[int], dom_offsets: List[int]):
        self.variable_nb = len(dom_indices)
        self.shr_domains = shr_domains
        self.shared_domains = self.build_shared_domains(shr_domains)
        self.domain_indices = self.build_domain_indices(dom_indices)
        self.domain_offsets = self.build_domain_offsets(dom_offsets)

    def reset(self) -> None:
        """
        Resets the shared domains to their initial values.
        """
        self.shared_domains = self.build_shared_domains(self.shr_domains)

    def build_shared_domains(self, shr_domains: List[Union[int, Tuple[int, int]]]) -> NDArray:
        return np.array(
            [(shr_domain, shr_domain) if isinstance(shr_domain, int) else shr_domain for shr_domain in shr_domains],
            dtype=np.int32,
            order="C",
        )

    def build_domain_indices(self, dom_indices: List[int]) -> NDArray:
        return np.array(dom_indices, dtype=np.uint16)

    def build_domain_offsets(self, dom_offsets: List[int]) -> NDArray:
        return np.array(dom_offsets, dtype=np.int32)

    def set_propagators(self, propagators: List[Tuple[List[int], int, List[int]]]) -> None:
        """
        Sets the propagators for the problem.
        The order of the propagators has an impact on the global speed of the resolution:
        it is recommended to put the more costly propagators at the end of the list.
        :param propagators: the list of propagators as tuples of the form (list of variables, algorithm, data).
        """
        self.prop_nb = len(propagators)
        prop_var_total_size = prop_data_total_size = 0
        self.prop_to_filter = np.empty(self.prop_nb, dtype=np.bool)
        self.prop_algorithms = np.empty(self.prop_nb, dtype=np.uint8)
        # We will store propagator specific data in a global arrays, we need to compute variables and data bounds.
        self.prop_var_bounds = np.empty(
            (self.prop_nb, 2), dtype=np.uint16, order="C"
        )  # there is a bit of redundancy here for faster access to the bounds
        self.prop_data_bounds = np.empty(
            (self.prop_nb, 2), dtype=np.uint16, order="C"
        )  # there is a bit of redundancy here for faster access to the bounds
        self.prop_var_bounds[0, START] = 0
        self.prop_data_bounds[0, START] = 0
        for pidx, propagator in enumerate(propagators):
            prop_var_size = len(propagator[0])
            prop_data_size = len(propagator[2])
            self.prop_algorithms[pidx] = propagator[1]
            if pidx > 0:
                self.prop_var_bounds[pidx, START] = self.prop_var_bounds[pidx - 1, END]
                self.prop_data_bounds[pidx, START] = self.prop_data_bounds[pidx - 1, END]
            self.prop_var_bounds[pidx, END] = self.prop_var_bounds[pidx, START] + prop_var_size
            self.prop_data_bounds[pidx, END] = self.prop_data_bounds[pidx, START] + prop_data_size
            prop_var_total_size += prop_var_size
            prop_data_total_size += prop_data_size
        # Bounds have been computed and can now be used.
        # The global arrays are the following:
        self.prop_dom_indices = np.empty(prop_var_total_size, dtype=np.uint16)
        self.prop_dom_offsets = np.empty(prop_var_total_size, dtype=np.int32)
        self.prop_triggers = np.empty((prop_var_total_size, 2), dtype=bool)
        self.prop_data = np.empty(prop_data_total_size, dtype=np.int32)
        # Let's init the global arrays.
        for pidx, propagator in enumerate(propagators):
            prop_vars = propagator[0]
            self.prop_dom_indices[self.prop_var_bounds[pidx, START] : self.prop_var_bounds[pidx, END]] = (
                self.domain_indices[prop_vars]
            )  # this is cached for faster access
            self.prop_dom_offsets[self.prop_var_bounds[pidx, START] : self.prop_var_bounds[pidx, END]] = (
                self.domain_offsets[prop_vars]
            )  # this is cached for faster access
            self.prop_data[self.prop_data_bounds[pidx, START] : self.prop_data_bounds[pidx, END]] = propagator[2]
            self.prop_triggers[self.prop_var_bounds[pidx, START] : self.prop_var_bounds[pidx, END]] = get_triggers(
                propagator[1], len(propagator[0]), propagator[2]
            )

    def get_values(self) -> List[int]:
        """
        Gets the values for the variables (when instantiated).
        :return: a list of integers
        """
        mins = self.shared_domains[self.domain_indices, MIN] + self.domain_offsets
        return mins.tolist()

    def set_min_value(self, var_idx: int, min_value: int) -> None:
        self.shared_domains[(self.domain_indices[var_idx]), MIN] = min_value - self.domain_offsets[var_idx]

    def set_max_value(self, var_idx: int, max_value: int) -> None:
        self.shared_domains[(self.domain_indices[var_idx]), MAX] = max_value - self.domain_offsets[var_idx]

    def __str__(self) -> str:
        return f"domains={self.shared_domains}"

    def filter(self, stats: NDArray = stats_init(), shr_dom_changes: Optional[NDArray] = None) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached.
        :param stats: where to record the statistics of the computation
        :param shr_dom_changes: an optional array of shared domain changes
        :return: False if the problem is not consistent
        """
        return filter(
            self.prop_nb,
            self.prop_to_filter,
            self.prop_algorithms,
            self.prop_var_bounds,
            self.prop_data_bounds,
            self.prop_dom_indices,
            self.prop_dom_offsets,
            self.prop_triggers,
            self.prop_data,
            self.shared_domains,
            stats,
            shr_dom_changes,
        )

    def pretty_print_solution(self, solution: List[int]) -> None:
        """
        Pretty prints a solution to the problem.
        :param solution: a list of integers
        """
        print(solution)


@jit(nopython=True, cache=True)
def filter(
    prop_nb: int,
    prop_to_filter: NDArray,
    prop_algorithms: NDArray,
    prop_var_bounds: NDArray,
    prop_data_bounds: NDArray,
    prop_indices: NDArray,
    prop_offsets: NDArray,
    prop_triggers: NDArray,
    prop_data: NDArray,
    shared_domains: NDArray,
    stats: NDArray,
    shr_dom_changes: Optional[NDArray],
) -> bool:
    """
    Filters the problem's domains by applying the propagators until a fix point is reached.
    :param stats: where to record the statistics of the computation
    :param shr_dom_changes: an optional array of shared domain changes
    :return: False if the problem is not consistent
    """
    stats[STATS_PROBLEM_FILTER_NB] += 1
    init_propagators_to_filter(prop_to_filter, shr_dom_changes, prop_nb, prop_var_bounds, prop_indices, prop_triggers)
    while True:
        # is there a propagator to filter ?
        none = True
        for pidx in range(prop_nb):
            if prop_to_filter[pidx]:
                none = prop_to_filter[pidx] = False
                break
        if none:
            return True
        # there is a propagator to filter
        stats[STATS_PROPAGATOR_FILTER_NB] += 1
        indices = prop_indices[prop_var_bounds[pidx, START] : prop_var_bounds[pidx, END]]
        offsets = prop_offsets[prop_var_bounds[pidx, START] : prop_var_bounds[pidx, END]].reshape((-1, 1))
        prop_domains = compute_domains(
            prop_algorithms[pidx],
            shared_domains[indices] + offsets,
            prop_data[prop_data_bounds[pidx, START] : prop_data_bounds[pidx, END]],
        )
        if prop_domains is None:
            return False
        old_shared_domains = shared_domains.copy()
        shared_domains[indices] = prop_domains - offsets
        update_propagators_to_filter(
            prop_to_filter,
            old_shared_domains != shared_domains,
            prop_nb,
            prop_var_bounds,
            prop_indices,
            prop_triggers,
            pidx,
        )


@jit(nopython=True, cache=True)
def is_solved(shared_domains: NDArray) -> bool:
    """
    Returns true iff the problem is solved.
    :return: a boolean
    """
    return bool(np.all(np.equal(shared_domains[:, MIN], shared_domains[:, MAX])))
