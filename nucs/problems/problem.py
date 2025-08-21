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
# Copyright 2024-2025 - Yan Georget
###############################################################################
import copy
import logging
from typing import Any, List, Optional, Self, Sequence, Tuple, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray
from rich import print

from nucs.constants import (
    EVENT_NB,
    NUMBA_DISABLE_JIT,
    PARAM,
    RANGE_END,
    RANGE_START,
    SIGNATURE_GET_TRIGGERS,
    TYPE_GET_TRIGGERS,
    VARIABLE,
)
from nucs.numba_helper import build_function_address_list, function_from_address
from nucs.propagators.propagators import GET_COMPLEXITY_FCTS, GET_TRIGGERS_FCTS

logger = logging.getLogger(__name__)


class Problem:
    """
    A problem is defined by:
    - a list of domains,
    - a list of propagators.
    A variable is a domain index.
    """

    def __init__(self, domains: Union[List[Tuple[int, int]], List[int]]):
        """
        Initializes the problem.
        :param domains: the domains expressed as a list
        """
        self.domains = [[domain, domain] if isinstance(domain, int) else [domain[0], domain[1]] for domain in domains]
        self.domain_nb = len(domains)
        self.propagators: List[Tuple[List[int], int, List[int]]] = []
        self.propagator_nb = 0

    def split(self, split_nb: int, var: int) -> List[Self]:
        """
        Splits a problem into several sub-problems by splitting the domain of a variable.
        :param split_nb: the number of sub-problems
        :param var: the index of the variable
        :return: a list of sub-problems
        """
        logger.debug(f"Splitting in {split_nb} problems with variable {var}")
        domain = self.domains[var]
        domain_min = domain[0]
        domain_max = domain[1]
        domain_size = domain_max - domain_min + 1
        problems = []
        min_idx = domain_min
        for split_idx in range(split_nb):
            problem = copy.deepcopy(self)
            max_idx = min_idx + domain_size // split_nb - (0 if split_idx < domain_size % split_nb else 1)
            problem.domains[var] = [min_idx, max_idx]
            min_idx = max_idx + 1
            problems.append(problem)
        return problems

    def add_variable(self, domain: Union[int, Tuple[int, int]]) -> int:
        """
        Adds an extra variable to the problem.
        :param domain: the domain of the variable
        :return: the extra variable
        """
        var = len(self.domains)
        self.domains.append([domain, domain] if isinstance(domain, int) else [domain[0], domain[1]])
        self.domain_nb = len(self.domains)
        return var

    def add_variables(self, domains: Sequence[Union[int, Tuple[int, int]]]) -> int:
        """
        Adds extra variables to the problem.
        :param domains: the domains of the variables
        :return: the first added variable
        """
        var = len(self.domains)
        self.domains.extend(
            [[domain, domain] if isinstance(domain, int) else [domain[0], domain[1]] for domain in domains]
        )
        self.domain_nb = len(self.domains)
        return var

    def add_propagator(self, algorithm: int, variables: List[int], parameters: Optional[List[int]] = None) -> None:
        """
        Adds an extra propagator.
        """
        if parameters is None:
            parameters = []
        self.propagators.append((variables, algorithm, parameters))
        self.propagator_nb = len(self.propagators)

    def init(self) -> None:
        """
        Completes the initialization of the problem.
        """
        logger.debug("Initializing problem")
        # Sort the propagators based on their estimated amortized complexities.
        self.propagators.sort(key=lambda prop: GET_COMPLEXITY_FCTS[prop[1]](len(prop[0]), prop[2]))
        self.algorithms = np.array([prop[1] for prop in self.propagators], dtype=np.uint8)
        # We will store propagator specific data in a global arrays, we need to compute variables and parameter bounds.
        logger.debug("Initializing bounds")
        self.bounds = np.zeros((max(1, self.propagator_nb), 2, 2), dtype=np.uint32)  # some redundancy here
        init_bounds(self.bounds, self.propagators)
        logger.debug("Initializing props")
        self.props_variables = np.empty(self.bounds[-1, VARIABLE, RANGE_END], dtype=np.uint32)
        self.props_parameters = np.empty(self.bounds[-1, PARAM, RANGE_END], dtype=np.int32)
        init_props(self.props_variables, self.props_parameters, self.bounds, self.propagators)
        logger.debug("Initializing triggers")
        self.triggers = np.full((self.domain_nb, 1 << EVENT_NB, self.propagator_nb + 1), -1, dtype=np.int32)
        get_triggers_addrs = (
            np.empty(0)
            if NUMBA_DISABLE_JIT
            else np.array(build_function_address_list(GET_TRIGGERS_FCTS, SIGNATURE_GET_TRIGGERS))
        )
        init_triggers(
            self.triggers,
            self.domain_nb,
            self.propagator_nb,
            self.bounds,
            self.props_variables,
            self.props_parameters,
            self.algorithms,
            get_triggers_addrs,
        )
        logger.debug("Problem initialized")
        logger.info(f"Problem has {self.propagator_nb} propagators")
        logger.info(f"Problem has {self.domain_nb} variables")

    def solution_as_printable(self, solution: NDArray) -> Any:
        return solution.tolist()

    def print_solution(self, solution: Optional[NDArray]) -> None:
        if solution is not None:
            print(self.solution_as_printable(solution))
        else:
            print("No solution")


def init_bounds(bounds: NDArray, propagators: List[Tuple[List[int], int, List[int]]]) -> None:
    for prop_idx, prop in enumerate(propagators):
        if prop_idx > 0:
            bounds[prop_idx, :, RANGE_START] = bounds[prop_idx - 1, :, RANGE_END]
        bounds[prop_idx, VARIABLE, RANGE_END] = bounds[prop_idx, VARIABLE, RANGE_START] + len(prop[0])
        bounds[prop_idx, PARAM, RANGE_END] = bounds[prop_idx, PARAM, RANGE_START] + len(prop[2])


def init_props(
    props_variables: NDArray,
    props_parameters: NDArray,
    bounds: NDArray,
    propagators: List[Tuple[List[int], int, List[int]]],
) -> None:
    for prop_idx, prop in enumerate(propagators):
        var_start = bounds[prop_idx, VARIABLE, RANGE_START]
        var_end = bounds[prop_idx, VARIABLE, RANGE_END]
        props_variables[var_start:var_end] = prop[0]
        param_start = bounds[prop_idx, PARAM, RANGE_START]
        param_end = bounds[prop_idx, PARAM, RANGE_END]
        props_parameters[param_start:param_end] = prop[2]


@njit(cache=True)
def init_triggers(
    triggers: NDArray,
    domain_nb: int,
    propagator_nb: int,
    bounds: NDArray,
    props_variables: NDArray,
    props_parameters: NDArray,
    algorithms: NDArray,
    get_triggers_addrs: NDArray,
) -> None:
    # for each domain and event, we store the list of propagator indices followed by -1
    indices = np.empty(domain_nb, dtype=np.int32)
    for event_mask in range(1, 1 << EVENT_NB):
        indices[:] = 0
        for prop_idx in range(propagator_nb):
            var_start = bounds[prop_idx, VARIABLE, RANGE_START]
            var_end = bounds[prop_idx, VARIABLE, RANGE_END]
            var_nb = var_end - var_start
            param_start = bounds[prop_idx, PARAM, RANGE_START]
            param_end = bounds[prop_idx, PARAM, RANGE_END]
            algorithm = algorithms[prop_idx]
            trigger_fct = (
                GET_TRIGGERS_FCTS[algorithm]
                if NUMBA_DISABLE_JIT
                else function_from_address(TYPE_GET_TRIGGERS, get_triggers_addrs[algorithm])
            )
            parameters = props_parameters[param_start:param_end]
            for var_idx in range(var_nb):
                if trigger_fct(var_nb, var_idx, parameters) & event_mask:
                    variable = props_variables[var_idx + var_start]
                    triggers[variable, event_mask, indices[variable]] = prop_idx
                    indices[variable] += 1
