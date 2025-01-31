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
from typing import Any, List, Optional, Self, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from rich import print

from nucs.constants import EVENT_NB, PARAM, RANGE_END, RANGE_START, VARIABLE
from nucs.propagators.propagators import GET_COMPLEXITY_FCTS, GET_TRIGGERS_FCTS

logger = logging.getLogger(__name__)


class Problem:
    """
    A problem is defined by:
    - a list of domains,
    - a list of variables and
    - a list of propagators.
    A variable references an offset and a domain.
    Domains can be shared between variables.
    """

    def __init__(
        self,
        domains: Union[List[Tuple[int, int]], List[int]],
        variables: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
    ):
        """
        Inits the problem.
        :param domains: the shared domains expressed as a list
        :param variables: the domain indices expressed as a list
        :param offsets: the domain offsets expressed as a list
        """
        n = len(domains)
        if variables is None:
            variables = list(range(0, n))
        if offsets is None:
            offsets = [0] * n
        self.domains = [[domain, domain] if isinstance(domain, int) else [domain[0], domain[1]] for domain in domains]
        self.variables = variables
        self.offsets = offsets
        self.domain_nb = len(domains)
        self.propagators: List[Tuple[List[int], int, List[int]]] = []
        self.propagator_nb = 0

    def split(self, split_nb: int, var_idx: int) -> List[Self]:
        """
        Splits a problem into several sub-problems by splitting the domain of a variable.
        :param split_nb: the number of sub-problems
        :param var_idx: the index of the variable
        :return: a list of sub-problems
        """
        logger.debug(f"Splitting in {split_nb} problems with variable {var_idx}")
        domain = self.domains[var_idx]
        domain_min = domain[0]
        domain_max = domain[1]
        domain_size = domain_max - domain_min + 1
        problems = []
        min_idx = domain_min
        for split_idx in range(split_nb):
            problem = copy.deepcopy(self)
            max_idx = min_idx + domain_size // split_nb - (0 if split_idx < domain_size % split_nb else 1)
            problem.domains[var_idx] = [min_idx, max_idx]
            min_idx = max_idx + 1
            problems.append(problem)
        return problems

    def add_variable(
        self, domain: Union[int, Tuple[int, int]], variable: Optional[int] = None, offset: Optional[int] = None
    ) -> int:
        """
        Adds an extra variable to the problem.
        :param domain: the shared domain of the variable
        :param variable: the domain index (automatically computed if not defined)
        :param offset: the domain offset (set to 0 if not defined)
        :return: the index of the extra variable
        """
        insertion_idx = len(self.domains)
        if variable is None:
            variable = insertion_idx
        if offset is None:
            offset = 0
        self.domains.append([domain, domain] if isinstance(domain, int) else [domain[0], domain[1]])
        self.variables.append(variable)
        self.offsets.append(offset)
        self.domain_nb = len(self.variables)
        return insertion_idx

    def add_variables(
        self,
        domains: List[Union[int, Tuple[int, int]]],
        variables: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
    ) -> int:
        """
        Adds extra variables to the problem.
        :param domains: the shared domains of the variables
        :param variables: the domain indices (automatically computed if not defined)
        :param offsets: the domain offsets (set to 0 if not defined)
        :return: the index where the extra variables have been added
        """
        insertion_idx = len(self.domains)
        n = len(domains)
        if variables is None:
            variables = [insertion_idx + i for i in range(n)]
        if offsets is None:
            offsets = [0] * n
        self.domains.extend(
            [[domain, domain] if isinstance(domain, int) else [domain[0], domain[1]] for domain in domains]
        )
        self.variables.extend(variables)
        self.offsets.extend(offsets)
        self.domain_nb = len(self.variables)
        return insertion_idx

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
        logger.debug("Initializing Problem")
        # Sort the propagators based on their estimated amortized complexities.
        self.propagators.sort(key=lambda prop: GET_COMPLEXITY_FCTS[prop[1]](len(prop[0]), prop[2]))
        # Variable and domain initialization
        self.variables_arr = np.array(self.variables, dtype=np.uint32)
        self.offsets_arr = np.array(self.offsets, dtype=np.int32)
        # Propagator initialization
        self.algorithms = np.empty(self.propagator_nb, dtype=np.uint8)
        # We will store propagator specific data in a global arrays, we need to compute variables and data bounds.
        bound_nb = max(1, self.propagator_nb)
        self.bounds = np.zeros((bound_nb, 2, 2), dtype=np.uint32)  # some redundancy here
        self.bounds[0, :, RANGE_START] = 0
        for prop_idx, prop in enumerate(self.propagators):
            prop_vars, prop_algorithm, prop_params = prop
            self.algorithms[prop_idx] = prop_algorithm
            if prop_idx > 0:
                self.bounds[prop_idx, :, RANGE_START] = self.bounds[prop_idx - 1, :, RANGE_END]
            self.bounds[prop_idx, VARIABLE, RANGE_END] = self.bounds[prop_idx, VARIABLE, RANGE_START] + len(prop_vars)
            self.bounds[prop_idx, PARAM, RANGE_END] = self.bounds[prop_idx, PARAM, RANGE_START] + len(prop_params)
        # Bounds have been computed and can now be used. The global arrays are the following:
        self.props_variables = np.empty(self.bounds[-1, VARIABLE, RANGE_END], dtype=np.uint32)
        for prop_idx, prop in enumerate(self.propagators):
            var_start = self.bounds[prop_idx, VARIABLE, RANGE_START]
            var_end = self.bounds[prop_idx, VARIABLE, RANGE_END]
            self.props_variables[var_start:var_end] = self.variables_arr[prop[0]]  # cached for faster access
        self.props_offsets = np.empty(self.bounds[-1, VARIABLE, RANGE_END], dtype=np.int32)
        for prop_idx, prop in enumerate(self.propagators):
            var_start = self.bounds[prop_idx, VARIABLE, RANGE_START]
            var_end = self.bounds[prop_idx, VARIABLE, RANGE_END]
            self.props_offsets[var_start:var_end] = self.offsets_arr[prop[0]]  # cached for faster access
        self.props_offsets = self.props_offsets.reshape((-1, 1))
        self.no_offsets = not np.any(self.offsets_arr)
        self.props_parameters = np.empty(self.bounds[-1, PARAM, RANGE_END], dtype=np.int32)
        for prop_idx, prop in enumerate(self.propagators):
            param_start = self.bounds[prop_idx, PARAM, RANGE_START]
            param_end = self.bounds[prop_idx, PARAM, RANGE_END]
            self.props_parameters[param_start:param_end] = prop[2]
        self.triggers = np.zeros((self.domain_nb, 1 << EVENT_NB, self.propagator_nb), dtype=np.bool)
        for prop_idx, prop in enumerate(self.propagators):
            prop_vars, prop_algorithm, prop_params = prop
            prop_triggers = GET_TRIGGERS_FCTS[prop_algorithm](len(prop_vars), prop_params)
            for prop_var_idx, prop_var in enumerate(prop_vars):
                for event_mask in range(1, 1 << EVENT_NB):
                    self.triggers[self.variables_arr[prop_var], event_mask, prop_idx] = prop_triggers[prop_var_idx] & event_mask
        logger.debug("Problem initialized")
        logger.info(f"Problem has {self.propagator_nb} propagators")
        logger.info(f"Problem has {self.domain_nb} variables")
        if self.no_offsets:
            logger.info("Problem does not use offsets")
        else:
            logger.info("Problem uses offsets")

    def solution_as_printable(self, solution: NDArray) -> Any:
        return solution.tolist()

    def print_solution(self, solution: Optional[NDArray]) -> None:
        if solution is not None:
            print(self.solution_as_printable(solution))
        else:
            print("No solution")
