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
# Copyright 2024-2026 - Yan Georget
###############################################################################
import copy
import logging
from typing import Any, Iterable, List, Optional, Self, Sequence, Tuple, Union

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray
from rich import print

from nucs.buckets import compute_priority
from nucs.constants import (
    EVENT_MASK_NB,
    NUMBA_DISABLE_JIT,
    PARAM,
    RANGE_END,
    RANGE_START,
    SIGN_GET_TRIGGERS,
    TYPE_GET_TRIGGERS,
    VARIABLE,
)
from nucs.numba_helper import addresses_from_functions, function_from_address
from nucs.propagators.propagators import GET_TRIGGERS_FCTS, GET_COMPLEXITY_FCTS

logger = logging.getLogger(__name__)


class Problem:
    """
    A problem is defined by:
    - a list of domains,
    - a list of propagators.
    A variable is a domain index.
    """

    def __init__(self, domains: Union[Iterable[Tuple[int, int]]]):
        """
        Initializes the problem.

        :param domains: the domains
        :type domains: Union[Iterable[Tuple[int, int]]]
        """
        self.unbound_variable_nb = 0
        self.domains = [(domain, domain) if isinstance(domain, int) else domain for domain in domains]
        self.domain_nb = len(self.domains)
        self.propagators: List[Tuple[List[int], int, List[int]]] = []
        self.propagator_nb = 0

    def split(self, split_nb: int, var: int) -> List[Self]:
        """
        Splits a problem into several sub-problems by splitting the domain of a variable.

        :param split_nb: the number of sub-problems
        :type split_nb: int
        :param var: the index of the variable
        :type var: int

        :return: a list of sub-problems
        :rtype: List[Self]
        """
        logger.debug(f"Splitting in {split_nb} problems with variable {var}")
        domain = self.domains[var]
        domain_min, domain_max = domain
        domain_size = domain_max - domain_min + 1
        problems = []
        min_idx = domain_min
        for split_idx in range(split_nb):
            problem = copy.deepcopy(self)
            max_idx = min_idx + domain_size // split_nb - (0 if split_idx < domain_size % split_nb else 1)
            problem.domains[var] = (min_idx, max_idx)
            min_idx = max_idx + 1
            problems.append(problem)
        return problems

    def add_variable(self, domain: Union[int, Tuple[int, int]]) -> int:
        """
        Adds an extra variable to the problem.

        :param domain: the domain of the variable
        :type domain: Union[int, Tuple[int, int]]

        :return: the extra variable
        :rtype: int
        """
        var = len(self.domains)
        self.domains.append((domain, domain) if isinstance(domain, int) else domain)
        self.domain_nb = var + 1
        return var

    def add_variables(self, domains: Sequence[Union[int, Tuple[int, int]]]) -> int:
        """
        Adds extra variables to the problem.

        :param domains: the domains of the variables
        :type domains: Sequence[Union[int, Tuple[int, int]]]

        :return: the first added variable
        :rtype: int
        """
        var = len(self.domains)
        self.domains.extend([(domain, domain) if isinstance(domain, int) else domain for domain in domains])
        self.domain_nb = len(self.domains)
        return var

    def add_propagator(
        self, algorithm: int, variables: Iterable[int], parameters: Optional[Iterable[int]] = None
    ) -> None:
        """
        Adds an extra propagator.

        :param algorithm: the algorithm id
        :type algorithm: int
        :param variables: the variables on which the propagator applies
        :type variables: Iterable[int]
        :param parameters: the parameters of the propagator
        :type parameters: Optional[Iterable[int]]
        """
        parameters = [] if parameters is None else list(parameters)
        variables = list(variables)
        self.propagators.append((variables, algorithm, parameters))
        self.propagator_nb += 1

    def init(self) -> None:
        """
        Completes the initialization of the problem.
        """
        logger.debug("Initializing problem")
        for domain_min, domain_max in self.domains:
            if domain_min != domain_max:
                self.unbound_variable_nb += 1
        self.algorithms = np.array([propagator[1] for propagator in self.propagators], dtype=np.uint8)
        # The propagation queue is a bucketed (priority) queue:
        # Priorities here store the bucket index = floor(log2(complexity)), clamped to [0, NB_BUCKETS-1].
        # Higher-complexitypropagators land in higher buckets and run after cheaper ones at fixpoint computation.
        self.priorities = np.array(
            [
                compute_priority(GET_COMPLEXITY_FCTS[propagator[1]](len(propagator[0]), propagator[2]))
                for propagator in self.propagators
            ],
            dtype=np.uint32,
        )
        # We will store propagator specific data in a global arrays, we need to compute variables and parameter bounds.
        logger.debug("Initializing bounds")
        self.bounds = np.zeros((max(1, self.propagator_nb), 2, 2), dtype=np.uint32)  # some redundancy here
        init_bounds(self.bounds, self.propagators)
        logger.debug("Initializing props")
        self.propagator_variables = np.empty(self.bounds[-1, VARIABLE, RANGE_END], dtype=np.uint32)
        self.propagator_parameters = np.empty(self.bounds[-1, PARAM, RANGE_END], dtype=np.int32)
        init_propagator_variables_and_parameters(
            self.propagator_variables, self.propagator_parameters, self.bounds, self.propagators
        )
        logger.debug("Initializing triggers")
        self.triggers = np.zeros((self.domain_nb, EVENT_MASK_NB, self.propagator_nb + 1), dtype=np.int32)
        init_triggers(
            self.triggers,
            self.domain_nb,
            self.propagator_nb,
            self.bounds,
            self.propagator_variables,
            self.propagator_parameters,
            self.algorithms,
            addresses_from_functions(GET_TRIGGERS_FCTS, SIGN_GET_TRIGGERS),
        )
        logger.debug("Problem initialized")
        logger.info(f"Problem has {self.propagator_nb} propagators")
        logger.info(f"Problem has {self.domain_nb} variables")

    def solution_as_printable(self, solution: NDArray) -> Any:
        """
        Returns a printable representation of a solution.

        :param solution: the solution
        :type solution: NDArray

        :return: a printable representation of the solution
        :rtype: Any
        """
        return solution.tolist()

    def print_solution(self, solution: Optional[NDArray]) -> None:
        """
        Prints a solution.

        :param solution: the solution, or None if there is no solution
        :type solution: Optional[NDArray]
        """
        print("No solution" if solution is None else self.solution_as_printable(solution))


def init_bounds(bounds: NDArray, propagators: List[Tuple[List[int], int, List[int]]]) -> None:
    """
    Initializes the variable and parameter bounds for each propagator.

    :param bounds: the bounds to initialize
    :type bounds: NDArray
    :param propagators: the propagators
    :type propagators: List[Tuple[List[int], int, List[int]]]
    """
    for propagator_idx, propagator in enumerate(propagators):
        if propagator_idx > 0:
            bounds[propagator_idx, :, RANGE_START] = bounds[propagator_idx - 1, :, RANGE_END]
        bounds[propagator_idx, VARIABLE, RANGE_END] = bounds[propagator_idx, VARIABLE, RANGE_START] + len(propagator[0])
        bounds[propagator_idx, PARAM, RANGE_END] = bounds[propagator_idx, PARAM, RANGE_START] + len(propagator[2])


def init_propagator_variables_and_parameters(
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    bounds: NDArray,
    propagators: List[Tuple[List[int], int, List[int]]],
) -> None:
    """
    Initializes the propagator variables and parameters arrays.

    :param propagator_variables: the propagator variables array to fill
    :type propagator_variables: NDArray
    :param propagator_parameters: the propagator parameters array to fill
    :type propagator_parameters: NDArray
    :param bounds: the bounds
    :type bounds: NDArray
    :param propagators: the propagators
    :type propagators: List[Tuple[List[int], int, List[int]]]
    """
    for propagator_idx, propagator in enumerate(propagators):
        var_start = bounds[propagator_idx, VARIABLE, RANGE_START]
        var_end = bounds[propagator_idx, VARIABLE, RANGE_END]
        propagator_variables[var_start:var_end] = propagator[0]
        param_start = bounds[propagator_idx, PARAM, RANGE_START]
        param_end = bounds[propagator_idx, PARAM, RANGE_END]
        propagator_parameters[param_start:param_end] = propagator[2]


@njit(cache=True, fastmath=True)
def init_triggers(
    triggers: NDArray,
    domain_nb: int,
    propagator_nb: int,
    bounds: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    algorithms: NDArray,
    get_triggers_addrs: NDArray,
) -> None:
    """
    Initializes the triggers array that maps each (variable, event) pair to the propagators to schedule.

    :param triggers: the triggers array to fill
    :type triggers: NDArray
    :param domain_nb: the number of domains
    :type domain_nb: int
    :param propagator_nb: the number of propagators
    :type propagator_nb: int
    :param bounds: the bounds
    :type bounds: NDArray
    :param propagator_variables: the propagator variables
    :type propagator_variables: NDArray
    :param propagator_parameters: the propagator parameters
    :type propagator_parameters: NDArray
    :param algorithms: the algorithm ids of the propagators
    :type algorithms: NDArray
    :param get_triggers_addrs: the addresses of the get_triggers functions
    :type get_triggers_addrs: NDArray
    """
    variable_propagator = np.full((domain_nb, propagator_nb), False)
    for propagator in range(propagator_nb):
        algorithm = algorithms[propagator]
        if NUMBA_DISABLE_JIT:
            trigger_fct = GET_TRIGGERS_FCTS[algorithm]
        else:
            trigger_fct = function_from_address(TYPE_GET_TRIGGERS, get_triggers_addrs[algorithm])  # type: ignore[call-arg, arg-type]
        parameters = propagator_parameters[
            bounds[propagator, PARAM, RANGE_START] : bounds[propagator, PARAM, RANGE_END]
        ]
        var_start = bounds[propagator, VARIABLE, RANGE_START]
        var_end = bounds[propagator, VARIABLE, RANGE_END]
        var_nb = var_end - var_start
        for var_idx in range(var_nb):
            variable = propagator_variables[var_start + var_idx]
            # beware, a propagator can have two variables corresponding to the same real variable
            if not variable_propagator[variable, propagator]:
                variable_propagator[variable, propagator] = True
                trigger = trigger_fct(var_nb, var_idx, parameters)
                for event_mask in range(1, EVENT_MASK_NB):
                    if trigger & event_mask:
                        triggers[variable, event_mask, 0] += 1
                        triggers[variable, event_mask, triggers[variable, event_mask, 0]] = propagator
