from typing import Tuple, Union

import numpy as np
from numba import jit  # type: ignore
from numba.typed import List
from numpy.typing import NDArray

from ncs.memory import (
    END,
    MAX,
    MIN,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    START,
    new_algorithms,
    new_bounds,
    new_data,
    new_domain_offsets_by_values,
    new_domains_by_values,
    new_entailed_propagators,
    new_indices,
    new_indices_by_values,
    new_offsets,
    new_triggered_propagators,
    new_triggers,
)
from ncs.propagators.propagators import (
    compute_domains,
    get_triggers,
    init_triggered_propagators,
    update_triggered_propagators,
)
from ncs.statistics import (
    STATS_PROBLEM_FILTER_NB,
    STATS_PROPAGATOR_ENTAILMENT_NB,
    STATS_PROPAGATOR_FILTER_NB,
    STATS_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_PROPAGATOR_INCONSISTENCY_NB,
    statistics_init,
)


class Problem:
    """
    A problem is conceptually defined by a list of domains, a list of variables and a list of propagators.
    A domain can be computed by applying an offset to a shared domain.
    """

    def __init__(self, shr_domains: List[Union[int, Tuple[int, int]]], dom_indices: List[int], dom_offsets: List[int]):
        """
        Inits the problem.
        :param shr_domains: the shared domains expressed as a list
        :param dom_indices: the domain indices expressed as a list
        :param dom_offsets: the domain offsets expressed as a list
        """
        self.variable_nb = len(dom_indices)
        self.shr_domains_backup = shr_domains  # a copy of the shared domains (as a list) used when resetting a problem
        self.shr_domains = new_domains_by_values(shr_domains)
        self.dom_indices = new_indices_by_values(dom_indices)
        self.dom_offsets = new_domain_offsets_by_values(dom_offsets)
        self.statistics = statistics_init()

    def reset_shr_domains(self) -> None:
        """
        Resets the shared domains to their initial values.
        """
        self.shr_domains = new_domains_by_values(self.shr_domains_backup)

    def reset_entailed_propagators(self) -> None:
        self.entailed_propagators[:] = False

    def set_propagators(self, propagators: List[Tuple[List[int], int, List[int]]]) -> None:
        """
        Completes the initialization of the problem by setting the propagators.
        The order of the propagators has an impact on the global speed of the resolution:
        it is recommended to put the more costly propagators at the end of the list.
        :param propagators: the list of propagators as tuples of the form (list of variables, algorithm, data).
        """
        self.propagator_nb = len(propagators)
        props_var_total_size = props_data_total_size = 0
        self.triggered_propagators = new_triggered_propagators(self.propagator_nb)
        self.entailed_propagators = new_entailed_propagators(self.propagator_nb)
        self.algorithms = new_algorithms(self.propagator_nb)
        # We will store propagator specific data in a global arrays, we need to compute variables and data bounds.
        self.var_bounds = new_bounds(
            self.propagator_nb
        )  # there is a bit of redundancy here for faster access to the bounds
        self.data_bounds = new_bounds(
            self.propagator_nb
        )  # there is a bit of redundancy here for faster access to the bounds
        self.var_bounds[0, START] = self.data_bounds[0, START] = 0
        for pidx, propagator in enumerate(propagators):
            prop_var_size = len(propagator[0])
            prop_data_size = len(propagator[2])
            self.algorithms[pidx] = propagator[1]
            if pidx > 0:
                self.var_bounds[pidx, START] = self.var_bounds[pidx - 1, END]
                self.data_bounds[pidx, START] = self.data_bounds[pidx - 1, END]
            self.var_bounds[pidx, END] = self.var_bounds[pidx, START] + prop_var_size
            self.data_bounds[pidx, END] = self.data_bounds[pidx, START] + prop_data_size
            props_var_total_size += prop_var_size
            props_data_total_size += prop_data_size
        # Bounds have been computed and can now be used.
        # The global arrays are the following:
        self.props_dom_indices = new_indices(props_var_total_size)
        self.props_dom_offsets = new_offsets(props_var_total_size)
        self.props_triggers = new_triggers(props_var_total_size, False)
        self.props_data = new_data(props_data_total_size)
        # Let's init the global arrays.
        for pidx, propagator in enumerate(propagators):
            prop_vars = propagator[0]
            self.props_dom_indices[self.var_bounds[pidx, START] : self.var_bounds[pidx, END]] = self.dom_indices[
                prop_vars
            ]  # this is cached for faster access
            self.props_dom_offsets[self.var_bounds[pidx, START] : self.var_bounds[pidx, END]] = self.dom_offsets[
                prop_vars
            ]  # this is cached for faster access
            self.props_data[self.data_bounds[pidx, START] : self.data_bounds[pidx, END]] = propagator[2]
            self.props_triggers[self.var_bounds[pidx, START] : self.var_bounds[pidx, END]] = get_triggers(
                propagator[1], len(propagator[0]), propagator[2]
            )

    def get_values(self) -> List[int]:
        """
        Gets the values for the variables (when instantiated).
        :return: a list of integers
        """
        mins = self.shr_domains[self.dom_indices, MIN] + self.dom_offsets
        return mins.tolist()

    def set_min_value(self, var_idx: int, min_value: int) -> None:
        self.shr_domains[(self.dom_indices[var_idx]), MIN] = min_value - self.dom_offsets[var_idx]

    def set_max_value(self, var_idx: int, max_value: int) -> None:
        self.shr_domains[(self.dom_indices[var_idx]), MAX] = max_value - self.dom_offsets[var_idx]

    def __str__(self) -> str:
        return f"domains={self.shr_domains}"

    def filter(self, shr_domain_changes: NDArray) -> bool:
        """
        Filters the problem's domains by applying the propagators until a fix point is reached.
        :param shr_domain_changes: an array of shared domain changes
        :return: False if the problem is not consistent
        """
        return filter(
            self.statistics,
            self.propagator_nb,
            self.triggered_propagators,
            self.entailed_propagators,
            self.algorithms,
            self.var_bounds,
            self.data_bounds,
            self.props_dom_indices,
            self.props_dom_offsets,
            self.props_triggers,
            self.props_data,
            self.shr_domains,
            shr_domain_changes,
        )

    def pretty_print_solution(self, solution: List[int]) -> None:
        """
        Pretty prints a solution to the problem.
        :param solution: a list of integers
        """
        print(solution)


@jit(nopython=True, cache=True)
def pop_propagator(propagator_nb: int, propagator_queue: NDArray) -> int:
    for prop_idx in range(propagator_nb):
        if propagator_queue[prop_idx]:
            propagator_queue[prop_idx] = False
            return prop_idx
    return -1


@jit(nopython=True, cache=True)
def filter(
    statistics: NDArray,
    propagator_nb: int,
    triggered_propagators: NDArray,
    entailed_propagators: NDArray,
    algorithms: NDArray,
    var_bounds: NDArray,
    data_bounds: NDArray,
    props_indices: NDArray,
    props_offsets: NDArray,
    props_triggers: NDArray,
    props_data: NDArray,
    shr_domains: NDArray,
    shr_domain_changes: NDArray,
) -> bool:
    """
    Filters the problem's domains by applying the propagators until a fix point is reached.
    :param shr_domain_changes: an array of shared domain changes
    :return: False if the problem is not consistent
    """
    shr_domains_cur = np.empty_like(shr_domains)
    statistics[STATS_PROBLEM_FILTER_NB] += 1
    init_triggered_propagators(
        triggered_propagators,
        entailed_propagators,
        shr_domain_changes,
        propagator_nb,
        var_bounds,
        props_indices,
        props_triggers,
    )
    while True:
        prop_idx = pop_propagator(propagator_nb, triggered_propagators)
        if prop_idx == -1:
            return True
        statistics[STATS_PROPAGATOR_FILTER_NB] += 1
        prop_var_start = var_bounds[prop_idx, START]
        prop_var_end = var_bounds[prop_idx, END]
        prop_indices = props_indices[prop_var_start:prop_var_end]
        prop_offsets = props_offsets[prop_var_start:prop_var_end].reshape((-1, 1))
        prop_domains = np.empty((2, len(prop_offsets)), dtype=np.int32).T  # trick for order=F
        np.add(shr_domains[prop_indices], prop_offsets, prop_domains)
        status = compute_domains(
            algorithms[prop_idx],
            prop_domains,
            props_data[data_bounds[prop_idx, START] : data_bounds[prop_idx, END]],
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_PROPAGATOR_INCONSISTENCY_NB] += 1
            return False
        if status == PROP_ENTAILMENT:
            entailed_propagators[prop_idx] = True
            statistics[STATS_PROPAGATOR_ENTAILMENT_NB] += 1
        shr_domains_cur[:, :] = shr_domains
        shr_domains[prop_indices] = prop_domains - prop_offsets
        shr_domain_changes[:, :] = shr_domains_cur != shr_domains
        if np.any(shr_domain_changes):  # type: ignore
            update_triggered_propagators(
                triggered_propagators,
                entailed_propagators,
                shr_domain_changes,
                propagator_nb,
                var_bounds,
                props_indices,
                props_triggers,
                prop_idx,
            )
        else:
            statistics[STATS_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1


@jit(nopython=True, cache=True)
def is_solved(shr_domains: NDArray) -> bool:
    """
    Returns true iff the problem is solved.
    :return: a boolean
    """
    return bool(np.all(np.equal(shr_domains[:, MIN], shr_domains[:, MAX])))
