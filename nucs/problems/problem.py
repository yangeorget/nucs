###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024 - Yan Georget
###############################################################################
import copy
from typing import Optional, Self, Tuple, Union

import numpy as np
from numba.typed import List

from nucs.constants import END, START
from nucs.propagators.propagators import GET_COMPLEXITY_FCTS, GET_TRIGGERS_FCTS


class Problem:
    """
    A problem is defined by a list of domains, a list of variables and a list of propagators.
    A domain is computed by adding an offset to a shared domain.
    """

    def __init__(
        self,
        shr_domains_lst: List[Union[int, Tuple[int, int]]],
        dom_indices_lst: Optional[List[int]] = None,
        dom_offsets_lst: Optional[List[int]] = None,
    ):
        """
        Inits the problem.
        :param shr_domains_lst: the shared domains expressed as a list
        :param dom_indices_lst: the domain indices expressed as a list
        :param dom_offsets_lst: the domain offsets expressed as a list
        """
        n = len(shr_domains_lst)
        if dom_indices_lst is None:
            dom_indices_lst = list(range(0, n))
        if dom_offsets_lst is None:
            dom_offsets_lst = [0] * n
        self.shr_domains_lst = [
            [domain, domain] if isinstance(domain, int) else [domain[0], domain[1]] for domain in shr_domains_lst
        ]
        self.dom_indices_lst = dom_indices_lst
        self.dom_offsets_lst = dom_offsets_lst
        self.variable_nb = len(shr_domains_lst)
        self.propagators: List[Tuple[List[int], int, List[int]]] = []
        self.propagator_nb = 0

    def split(self, split_nb: int, var_idx: int) -> List[Self]:
        """
        Splits a problem into several sub-problems by splitting the domain of a variable.
        :param split_nb: the number of sub-problems
        :param var_idx: the index of the variable
        :return: a list of sub-problems
        """
        shr_dom = self.shr_domains_lst[var_idx]
        shr_dom_min = shr_dom[0]
        shr_dom_max = shr_dom[1]
        shr_dom_sz = shr_dom_max - shr_dom_min + 1
        problems = []
        min_idx = shr_dom_min
        for split_idx in range(split_nb):
            problem = copy.deepcopy(self)
            max_idx = min_idx + shr_dom_sz // split_nb - (0 if split_idx < shr_dom_sz % split_nb else 1)
            problem.shr_domains_lst[var_idx] = [min_idx, max_idx]
            min_idx = max_idx + 1
            problems.append(problem)
        return problems

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
        self.shr_domains_lst.append(
            [shr_domain, shr_domain] if isinstance(shr_domain, int) else [shr_domain[0], shr_domain[1]]
        )
        self.dom_indices_lst.append(dom_index)
        self.dom_offsets_lst.append(dom_offset)
        self.variable_nb = len(self.dom_indices_lst)

    def add_variables(
        self,
        shr_domains_list: List[Union[int, Tuple[int, int]]],
        dom_indices_list: Optional[List[int]] = None,
        dom_offsets_list: Optional[List[int]] = None,
    ) -> None:
        """
        Adds extra variables to the problem.
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
        self.shr_domains_lst.extend(
            [
                [shr_domain, shr_domain] if isinstance(shr_domain, int) else [shr_domain[0], shr_domain[1]]
                for shr_domain in shr_domains_list
            ]
        )
        self.dom_indices_lst.extend(dom_indices_list)
        self.dom_offsets_lst.extend(dom_offsets_list)
        self.variable_nb = len(self.dom_indices_lst)

    def add_propagator(self, propagator: Tuple[List[int], int, List[int]]) -> None:
        """
        Adds an extra propagator.
        :param propagator: the propagator
        """
        self.propagators.append(propagator)
        self.propagator_nb = len(self.propagators)

    def add_propagators(self, propagators: List[Tuple[List[int], int, List[int]]]) -> None:
        """
        Adds a list of propagators
        :param propagators: the propagators
        """
        self.propagators.extend(propagators)
        self.propagator_nb = len(self.propagators)

    def init(self) -> None:
        """
        Completes the initialization of the problem.
        """
        # Sort the propagators based on their estimated amortized complexities.
        self.propagators.sort(key=lambda prop: GET_COMPLEXITY_FCTS[prop[1]](len(prop[0]), prop[2]))
        # Variable and domain initialization
        self.dom_indices_arr = np.array(self.dom_indices_lst, dtype=np.uint16)
        self.dom_offsets_arr = np.array(self.dom_offsets_lst, dtype=np.int32)
        # Propagator initialization
        self.algorithms = np.empty(self.propagator_nb, dtype=np.uint8)
        # We will store propagator specific data in a global arrays, we need to compute variables and data bounds.
        bound_nb = max(1, self.propagator_nb)
        self.var_bounds = np.zeros((bound_nb, 2), dtype=np.uint16)  # some redundancy here
        self.param_bounds = np.zeros((bound_nb, 2), dtype=np.uint16)  # some redundancy here
        self.var_bounds[0, START] = self.param_bounds[0, START] = 0
        for propagator_idx, propagator in enumerate(self.propagators):
            prop_vars, prop_algorithm, prop_params = propagator
            self.algorithms[propagator_idx] = prop_algorithm
            if propagator_idx > 0:
                self.var_bounds[propagator_idx, START] = self.var_bounds[propagator_idx - 1, END]
                self.param_bounds[propagator_idx, START] = self.param_bounds[propagator_idx - 1, END]
            self.var_bounds[propagator_idx, END] = self.var_bounds[propagator_idx, START] + len(prop_vars)
            self.param_bounds[propagator_idx, END] = self.param_bounds[propagator_idx, START] + len(prop_params)
        # Bounds have been computed and can now be used. The global arrays are the following:
        self.props_dom_indices = np.empty(self.var_bounds[-1, END], dtype=np.uint16)
        self.props_dom_offsets = np.empty(self.var_bounds[-1, END], dtype=np.int32)
        self.props_parameters = np.empty(self.param_bounds[-1, END], dtype=np.int32)
        for propagator_idx, propagator in enumerate(self.propagators):
            prop_vars, _, prop_params = propagator
            var_start = self.var_bounds[propagator_idx, START]
            var_end = self.var_bounds[propagator_idx, END]
            self.props_dom_indices[var_start:var_end] = self.dom_indices_arr[prop_vars]  # cached for faster access
            self.props_dom_offsets[var_start:var_end] = self.dom_offsets_arr[prop_vars]  # cached for faster access
            param_start = self.param_bounds[propagator_idx, START]
            param_end = self.param_bounds[propagator_idx, END]
            self.props_parameters[param_start:param_end] = prop_params
        self.props_dom_offsets = self.props_dom_offsets.reshape((-1, 1))
        self.shr_domains_propagators = np.zeros((self.variable_nb, 2, self.propagator_nb), dtype=np.bool)
        for propagator_idx, propagator in enumerate(self.propagators):
            prop_vars, prop_algorithm, prop_params = propagator
            triggers = GET_TRIGGERS_FCTS[prop_algorithm](len(prop_vars), prop_params)
            for prop_var_idx, prop_var in enumerate(prop_vars):
                self.shr_domains_propagators[self.dom_indices_arr[prop_var], :, propagator_idx] = triggers[
                    prop_var_idx, :
                ]
