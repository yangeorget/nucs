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
import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray
from rich import print

from nucs.buckets import compute_priority
from nucs.constants import EVENT_MASK_NB, EVENT_NB
from nucs.numba_helper import NUMBA_DISABLE_JIT, addresses_from_functions, function_ptr_from_address
from nucs.propagators.propagators import (
    ALG_DUMMY,
    GET_COMPLEXITY_FCTS,
    GET_TRIGGERS_FCTS,
    IDEMPOTENCIES,
    IS_VACUOUS_FCTS,
    SIGN_GET_TRIGGERS,
    TYPE_GET_TRIGGERS,
)

logger = logging.getLogger(__name__)

PROBLEM_INCONSISTENT = 0  # returned when the filtering of a problem detects an inconsistency
PROBLEM_UNBOUND = 1  # returned when the filtering of a problem has been completed, but the problem is not solved
PROBLEM_BOUND = 2  # returned when a problem is solved

# Offsets columns
OFFSETS_VARIABLE = 0  # column of offsets holding the propagator variable offsets
OFFSETS_PARAM = 1  # column of offsets holding the propagator parameter offsets


class Problem:
    """
    A problem is defined by:
    - a list of domains,
    - a list of propagators.
    A variable is a domain index.
    """

    def __init__(self, domains: Iterable[tuple[int, int]]):
        """
        Initializes the problem.

        :param domains: the domains
        :type domains: Union[Iterable[Tuple[int, int]]]
        """
        self.unbound_variable_nb = 0
        self.domains = [(domain, domain) if isinstance(domain, int) else domain for domain in domains]
        self.domain_nb = len(self.domains)
        self.propagators: list[tuple[list[int], int, list[int]]] = []
        self.propagator_nb = 0

    def add_variable(self, domain: int | tuple[int, int]) -> int:
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

    def add_variables(self, domains: Sequence[int | tuple[int, int]]) -> int:
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

    def add_propagator(self, algorithm: int, variables: Iterable[int], parameters: Iterable[int] | None = None) -> None:
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
        # Some constraints are made vacuous by their parameters alone: no assignment can violate them, whatever
        # the domains. That is a property of the model, not of the search, so it is settled once here rather
        # than re-derived at every subtree root: the propagator is simply not posted, which costs no call, no
        # entry in the trigger buckets and no slot in the propagator arrays.
        if IS_VACUOUS_FCTS[algorithm](len(variables), parameters, [self.domains[variable] for variable in variables]):
            logger.debug(f"Not posting a vacuous propagator {algorithm}")
            return
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
        # the compiled form of the domains the model was built with. domains stays the list the model API
        # appends to; this is what a search resets to, and it has to survive the search that overwrites the
        # solver's own domains -- those are a view of state. int32 because that is what it is copied into.
        self.initial_domains = np.array(self.domains, dtype=np.int32)
        self.algorithms = np.array([propagator[1] for propagator in self.propagators], dtype=np.uint8)
        # Built here, beside the algorithms that index it, so the two are fixed at the same instant and the
        # flags cover every algorithm this problem can name -- including one registered after import. The
        # consistency algorithm needs a boolean array rather than the registry's list; converting some sixty
        # bools once per problem is nothing next to the rest of this method.
        self.idempotencies = np.array(IDEMPOTENCIES, dtype=np.bool)
        self.init_priorities()
        self.init_propagator_arrays()
        self.init_triggers()
        logger.debug("Problem initialized")
        logger.info(f"Problem has {self.propagator_nb} propagators")
        logger.info(f"Problem has {self.domain_nb} variables")

    def init_priorities(self) -> None:
        """
        Initializes the priorities of the propagators.

        The propagation queue is a bucketed (priority) queue: priorities here store the bucket index =
        floor(log2(complexity)), clamped to [0, NB_BUCKETS-1]. Higher-complexity propagators land in higher
        buckets and run after cheaper ones at fixpoint computation.
        """
        logger.debug("Initializing priorities")
        self.priorities = np.array(
            [
                compute_priority(GET_COMPLEXITY_FCTS[propagator[1]](len(propagator[0]), propagator[2]))
                for propagator in self.propagators
            ],
            dtype=np.uint32,
        )

    def init_propagator_arrays(self) -> None:
        """
        Initializes the offsets and the propagator variables and parameters.

        Propagator specific data lives in global arrays; propagator p owns the slice
        offsets[p, col]:offsets[p + 1, col] of each. The slices are contiguous, so one offset per propagator
        suffices: one row per propagator plus a closing row holding the totals.
        """
        logger.debug("Initializing offsets")
        self.offsets = np.zeros((self.propagator_nb + 1, 2), dtype=np.uint32)
        for propagator_idx, propagator in enumerate(self.propagators):
            self.offsets[propagator_idx + 1, OFFSETS_VARIABLE] = self.offsets[propagator_idx, OFFSETS_VARIABLE] + len(
                propagator[0]
            )
            self.offsets[propagator_idx + 1, OFFSETS_PARAM] = self.offsets[propagator_idx, OFFSETS_PARAM] + len(
                propagator[2]
            )
        logger.debug("Initializing props")
        self.propagator_variables = np.empty(self.offsets[-1, OFFSETS_VARIABLE], dtype=np.uint32)
        self.propagator_parameters = np.empty(self.offsets[-1, OFFSETS_PARAM], dtype=np.int32)
        for propagator_idx, propagator in enumerate(self.propagators):
            var_start, var_end = (
                self.offsets[propagator_idx, OFFSETS_VARIABLE],
                self.offsets[propagator_idx + 1, OFFSETS_VARIABLE],
            )
            param_start, param_end = (
                self.offsets[propagator_idx, OFFSETS_PARAM],
                self.offsets[propagator_idx + 1, OFFSETS_PARAM],
            )
            self.propagator_variables[var_start:var_end] = propagator[0]
            self.propagator_parameters[param_start:param_end] = propagator[2]

    def init_triggers(self) -> None:
        """
        Initializes the triggers, mapping each (variable, event) pair to the propagators to schedule.

        A dense (domain_nb, EVENT_MASK_NB, propagator_nb) array would be mostly empty (and huge), so the map
        is stored in CSR form: triggers is the flat list of propagators and triggers_offsets delimits, for
        each (variable, event), its slice -- offsets[variable * EVENT_MASK_NB + event] up to the next offset.

        Requires the algorithms, the offsets and the propagator variables and parameters to be initialized.
        """
        logger.debug("Initializing triggers")
        # resolving only the algorithms used by the problem keeps the init cost proportional
        # to the problem instead of the whole propagator library
        get_triggers_addrs = addresses_from_functions(
            GET_TRIGGERS_FCTS, SIGN_GET_TRIGGERS, np.unique(self.algorithms), ALG_DUMMY
        )
        counts = np.zeros((self.domain_nb, EVENT_MASK_NB), dtype=np.int32)
        count_triggers(
            counts,
            self.offsets,
            self.propagator_variables,
            self.propagator_parameters,
            self.algorithms,
            get_triggers_addrs,
        )
        self.triggers_offsets = np.zeros((self.domain_nb << EVENT_NB) + 1, dtype=np.int32)
        np.cumsum(counts.reshape(-1), out=self.triggers_offsets[1:])
        self.triggers = np.empty(int(self.triggers_offsets[-1]), dtype=np.int32)
        cursors = self.triggers_offsets[:-1].copy()
        fill_triggers(
            self.triggers,
            cursors,
            self.offsets,
            self.propagator_variables,
            self.propagator_parameters,
            self.algorithms,
            get_triggers_addrs,
        )

    def solution_as_printable(self, solution: NDArray) -> Any:
        """
        Returns a printable representation of a solution.

        :param solution: the solution
        :type solution: NDArray

        :return: a printable representation of the solution
        :rtype: Any
        """
        return solution.tolist()

    def print_solution(self, solution: NDArray | None) -> None:
        """
        Prints a solution.

        :param solution: the solution, or None if there is no solution
        :type solution: Optional[NDArray]
        """
        print("No solution" if solution is None else self.solution_as_printable(solution))


@njit(cache=True)
def count_triggers(
    counts: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    algorithms: NDArray,
    get_triggers_addrs: NDArray,
) -> None:
    """
    Counts, for each (variable, event) pair, the number of propagators to schedule, sizing the CSR triggers.

    :param counts: the (domain_nb, EVENT_MASK_NB) array of counts to fill
    :type counts: NDArray
    :param offsets: the CSR offsets delimiting each propagator's variables and parameters
    :type offsets: NDArray
    :param propagator_variables: the propagator variables
    :type propagator_variables: NDArray
    :param propagator_parameters: the propagator parameters
    :type propagator_parameters: NDArray
    :param algorithms: the algorithm ids of the propagators
    :type algorithms: NDArray
    :param get_triggers_addrs: the addresses of the get_triggers functions
    :type get_triggers_addrs: NDArray
    """
    for propagator in range(len(algorithms)):
        algorithm = algorithms[propagator]
        if NUMBA_DISABLE_JIT:
            trigger_fct = GET_TRIGGERS_FCTS[algorithm]
        else:
            trigger_fct = function_ptr_from_address(TYPE_GET_TRIGGERS, get_triggers_addrs[algorithm])  # type: ignore[call-arg, arg-type]
        parameters = propagator_parameters[offsets[propagator, OFFSETS_PARAM] : offsets[propagator + 1, OFFSETS_PARAM]]
        var_start = offsets[propagator, OFFSETS_VARIABLE]
        var_end = offsets[propagator + 1, OFFSETS_VARIABLE]
        var_nb = var_end - var_start
        for var_idx in range(var_nb):
            variable = propagator_variables[var_start + var_idx]
            # beware, a propagator can reference the same variable twice; only the first occurrence counts
            duplicate = False
            for prev in range(var_idx):
                if propagator_variables[var_start + prev] == variable:
                    duplicate = True
                    break
            if duplicate:
                continue
            trigger = trigger_fct(var_nb, var_idx, parameters)
            for event_mask in range(1, EVENT_MASK_NB):
                if trigger & event_mask:
                    counts[variable, event_mask] += 1


@njit(cache=True)
def fill_triggers(
    triggers: NDArray,
    cursors: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    algorithms: NDArray,
    get_triggers_addrs: NDArray,
) -> None:
    """
    Fills the flat CSR triggers array, writing each propagator into the slice of every (variable, event) it
    triggers; cursors holds, per (variable, event), the next write position (initialized to the slice start).

    :param triggers: the flat triggers array to fill
    :type triggers: NDArray
    :param cursors: the per (variable, event) write cursors, indexed variable * EVENT_MASK_NB + event
    :type cursors: NDArray
    :param offsets: the CSR offsets delimiting each propagator's variables and parameters
    :type offsets: NDArray
    :param propagator_variables: the propagator variables
    :type propagator_variables: NDArray
    :param propagator_parameters: the propagator parameters
    :type propagator_parameters: NDArray
    :param algorithms: the algorithm ids of the propagators
    :type algorithms: NDArray
    :param get_triggers_addrs: the addresses of the get_triggers functions
    :type get_triggers_addrs: NDArray
    """
    for propagator in range(len(algorithms)):
        algorithm = algorithms[propagator]
        if NUMBA_DISABLE_JIT:
            trigger_fct = GET_TRIGGERS_FCTS[algorithm]
        else:
            trigger_fct = function_ptr_from_address(TYPE_GET_TRIGGERS, get_triggers_addrs[algorithm])  # type: ignore[call-arg, arg-type]
        parameters = propagator_parameters[offsets[propagator, OFFSETS_PARAM] : offsets[propagator + 1, OFFSETS_PARAM]]
        var_start = offsets[propagator, OFFSETS_VARIABLE]
        var_end = offsets[propagator + 1, OFFSETS_VARIABLE]
        var_nb = var_end - var_start
        for var_idx in range(var_nb):
            variable = propagator_variables[var_start + var_idx]
            duplicate = False
            for prev in range(var_idx):
                if propagator_variables[var_start + prev] == variable:
                    duplicate = True
                    break
            if duplicate:
                continue
            trigger = trigger_fct(var_nb, var_idx, parameters)
            for event_mask in range(1, EVENT_MASK_NB):
                if trigger & event_mask:
                    position = cursors[(variable << EVENT_NB) | event_mask]
                    triggers[position] = propagator
                    cursors[(variable << EVENT_NB) | event_mask] = position + 1
