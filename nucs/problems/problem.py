from typing import Optional, Tuple, Union

import numpy as np
from numba import njit  # type: ignore
from numba.typed import List
from numpy.typing import NDArray

from nucs.constants import END, MAX, MIN, START
from nucs.numpy import (
    new_algorithms,
    new_bounds,
    new_data,
    new_dom_indices,
    new_dom_indices_by_values,
    new_dom_offsets,
    new_dom_offsets_by_values,
    new_not_entailed_propagators,
    new_shr_domains_by_values,
    new_shr_domains_propagators,
    new_triggered_propagators,
)
from nucs.propagators.propagators import GET_COMPLEXITY_FCTS, GET_TRIGGERS_FCTS
from nucs.statistics import STATS_PROBLEM_PROPAGATOR_NB, STATS_PROBLEM_VARIABLE_NB


class Problem:
    """
    A problem is conceptually defined by a list of domains, a list of variables and a list of propagators.
    A domain can be computed by applying an offset to a shared domain.
    """

    def __init__(
        self,
        shr_domains_list: List[Union[int, Tuple[int, int]]],
        dom_indices_list: Optional[List[int]] = None,
        dom_offsets_list: Optional[List[int]] = None,
    ):
        """
        Inits the problem.
        :param shr_domains_list: the shared domains expressed as a list
        :param dom_indices_list: the domain indices expressed as a list
        :param dom_offsets_list: the domain offsets expressed as a list
        """
        n = len(shr_domains_list)
        if dom_indices_list is None:
            dom_indices_list = list(range(0, n))
        if dom_offsets_list is None:
            dom_offsets_list = [0] * n
        self.shr_domains_lst = shr_domains_list
        self.dom_indices_lst = dom_indices_list
        self.dom_offsets_lst = dom_offsets_list
        self.propagators: List[Tuple[List[int], int, List[int]]] = []
        self.propagator_nb = 0
        self.ready = False  # the problem is not yet ready to be used, init_problem() must be called

    def add_variable(
        self, shr_domain: Union[int, Tuple[int, int]], dom_index: Optional[int] = None, dom_offset: Optional[int] = None
    ) -> None:
        """
        Adds an extra variable to the problem.
        :param shr_domain: the shared domain of the variable
        :param dom_index: the domain index (automatically computed if not defined)
        :param dom_offset: the domain offset (set to 0 if not defined)
        """
        start = len(self.shr_domains_lst)
        if dom_index is None:
            dom_index = start
        if dom_offset is None:
            dom_offset = 0
        self.shr_domains_lst.append(shr_domain)
        self.dom_indices_lst.append(dom_index)
        self.dom_offsets_lst.append(dom_offset)

    def add_variables(
        self,
        shr_domains_list: List[Union[int, Tuple[int, int]]],
        dom_indices_list: Optional[List[int]] = None,
        dom_offsets_list: Optional[List[int]] = None,
    ) -> None:
        """
        Adds extra variabled to the problem.
        :param shr_domains_list: the shared domains of the variables
        :param dom_indices_list: the domain indices (automatically computed if not defined)
        :param dom_offsets_list: the domain offsets (set to 0 if not defined)
        """
        start = len(self.shr_domains_lst)
        n = len(shr_domains_list)
        if dom_indices_list is None:
            dom_indices_list = [start + i for i in range(n)]
        if dom_offsets_list is None:
            dom_offsets_list = [0] * n
        self.shr_domains_lst.extend(shr_domains_list)
        self.dom_indices_lst.extend(dom_indices_list)
        self.dom_offsets_lst.extend(dom_offsets_list)

    def add_propagator(self, propagator: Tuple[List[int], int, List[int]], pos: int = -1) -> None:
        """
        Adds an extra propagator.
        :param propagator: the propagator
        """
        if pos == -1:
            self.propagators.append(propagator)
        else:
            self.propagators.insert(pos, propagator)

    def add_propagators(self, propagators: List[Tuple[List[int], int, List[int]]]) -> None:
        """
        Adds a list of propagators
        :param propagators: the propagators
        """
        self.propagators.extend(propagators)

    def init_problem(self, statistics: Optional[NDArray] = None) -> None:
        """
        Completes the initialization of the problem by defining the variables and the propagators.
        """
        # Variable and domain initialization
        self.variable_nb = len(self.dom_indices_lst)
        self.shr_domains_arr = new_shr_domains_by_values(self.shr_domains_lst)
        self.dom_indices_arr = new_dom_indices_by_values(self.dom_indices_lst)
        self.dom_offsets_arr = new_dom_offsets_by_values(self.dom_offsets_lst)
        # Sort the propagators based on their estimated amortized complexities.
        self.propagators.sort(key=lambda prop: GET_COMPLEXITY_FCTS[prop[1]](len(prop[0]), prop[2]))
        # Propagator initialization
        self.propagator_nb = len(self.propagators)
        # This is where the triggered propagators will be stored,
        # propagators will be computed in the order of their increasing indices.
        # This is empty at the end of a filter.
        self.triggered_propagators = new_triggered_propagators(self.propagator_nb)
        # This is where the entailed propagators will be stored
        # This is reset in the case of braktrack
        self.not_entailed_propagators = new_not_entailed_propagators(self.propagator_nb)
        self.algorithms = new_algorithms(self.propagator_nb)
        # We will store propagator specific data in a global arrays, we need to compute variables and data bounds.
        self.var_bounds = new_bounds(max(1, self.propagator_nb))  # some redundancy here
        self.data_bounds = new_bounds(max(1, self.propagator_nb))  # some redundancy here
        self.var_bounds[0, START] = self.data_bounds[0, START] = 0
        for pidx, propagator in enumerate(self.propagators):
            self.algorithms[pidx] = propagator[1]
            if pidx > 0:
                self.var_bounds[pidx, START] = self.var_bounds[pidx - 1, END]
                self.data_bounds[pidx, START] = self.data_bounds[pidx - 1, END]
            self.var_bounds[pidx, END] = self.var_bounds[pidx, START] + len(propagator[0])
            self.data_bounds[pidx, END] = self.data_bounds[pidx, START] + len(propagator[2])
        # Bounds have been computed and can now be used. The global arrays are the following:
        self.props_dom_indices = new_dom_indices(self.var_bounds[-1, END])
        self.props_dom_offsets = new_dom_offsets(self.var_bounds[-1, END])
        self.props_data = new_data(self.data_bounds[-1, END])
        for pidx, propagator in enumerate(self.propagators):
            prop_vars = propagator[0]
            self.props_dom_indices[self.var_bounds[pidx, START] : self.var_bounds[pidx, END]] = self.dom_indices_arr[
                prop_vars
            ]  # this is cached for faster access
            self.props_dom_offsets[self.var_bounds[pidx, START] : self.var_bounds[pidx, END]] = self.dom_offsets_arr[
                prop_vars
            ]  # this is cached for faster access
            self.props_data[self.data_bounds[pidx, START] : self.data_bounds[pidx, END]] = propagator[2]
        self.props_dom_offsets = self.props_dom_offsets.reshape((-1, 1))
        self.shr_domains_propagators = new_shr_domains_propagators(len(self.shr_domains_lst), self.propagator_nb)
        for propagator_idx, propagator in enumerate(self.propagators):
            triggers = GET_TRIGGERS_FCTS[propagator[1]](len(propagator[0]), propagator[2])
            for prop_variable_idx, prop_variable in enumerate(propagator[0]):
                self.shr_domains_propagators[self.dom_indices_arr[prop_variable], :, propagator_idx] = triggers[
                    prop_variable_idx, :
                ]
        if statistics is not None:
            statistics[STATS_PROBLEM_PROPAGATOR_NB] = self.propagator_nb
            statistics[STATS_PROBLEM_VARIABLE_NB] = self.variable_nb

    def reset(self, new_shr_domains_arr: Optional[NDArray] = None) -> None:
        self.shr_domains_arr = (
            new_shr_domains_arr if new_shr_domains_arr is not None else new_shr_domains_by_values(self.shr_domains_lst)
        )
        self.triggered_propagators.fill(True)
        self.not_entailed_propagators.fill(True)

    def get_min_value(self, var_idx: int) -> int:
        """
        Gets the minimal value of a variable.
        :param var_idx: the index of the variable
        :return: the minimal value
        """
        return self.shr_domains_arr[self.dom_indices_arr[var_idx], MIN] + self.dom_offsets_arr[var_idx]

    def get_max_value(self, var_idx: int) -> int:
        """
        Gets the maximal value of a variable.
        :param var_idx: the index of the variable
        :return: the maximal value
        """
        return self.shr_domains_arr[self.dom_indices_arr[var_idx], MAX] + self.dom_offsets_arr[var_idx]

    def set_min_value(self, var_idx: int, min_value: int) -> None:
        """
        Sets the minimal value of a variable.
        :param var_idx: the index of the variable
        :param min_value: the minimal value
        """
        self.shr_domains_arr[self.dom_indices_arr[var_idx], MIN] = min_value - self.dom_offsets_arr[var_idx]

    def set_max_value(self, var_idx: int, max_value: int) -> None:
        """
        Sets the maximal value of a variable.
        :param var_idx: the index of the variable
        :param min_value: the maximal value
        """
        self.shr_domains_arr[self.dom_indices_arr[var_idx], MAX] = max_value - self.dom_offsets_arr[var_idx]

    def __str__(self) -> str:
        return f"domains={self.shr_domains_arr}, indices={self.dom_indices_arr}, offsets={self.dom_offsets_arr}"

    def pretty_print_solution(self, solution: List[int]) -> None:
        """
        Pretty prints a solution to the problem.
        :param solution: a list of integers
        """
        print(solution)


@njit(cache=True)
def is_solved(shr_domains: NDArray) -> bool:
    """
    Returns true iff the problem is solved.
    :return: a boolean
    """
    return bool(np.all(np.equal(shr_domains[:, MIN], shr_domains[:, MAX])))
