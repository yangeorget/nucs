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
import logging
from multiprocessing import Queue
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    LOG_LEVEL_INFO,
    NUMBA_DISABLE_JIT,
    PROBLEM_BOUND,
    PROBLEM_UNBOUND,
    SIGNATURE_COMPUTE_DOMAINS,
    SIGNATURE_CONSISTENCY_ALG,
    SIGNATURE_DOM_HEURISTIC,
    SIGNATURE_VAR_HEURISTIC,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_ALG_BC_WITH_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_CHANGE_NB,
    STATS_IDX_ALG_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_SOLUTION_NB,
    STATS_LBL_ALG_BC_NB,
    STATS_LBL_ALG_BC_WITH_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_CHANGE_NB,
    STATS_LBL_ALG_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_ENTAILMENT_NB,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_INCONSISTENCY_NB,
    STATS_LBL_SOLVER_BACKTRACK_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
    STATS_LBL_SOLVER_CHOICE_NB,
    STATS_LBL_SOLVER_SOLUTION_NB,
    STATS_MAX,
    TYPE_CONSISTENCY_ALG,
    TYPE_DOM_HEURISTIC,
    TYPE_VAR_HEURISTIC,
)
from nucs.numba_helper import build_function_address_list, function_from_address
from nucs.problems.problem import Problem
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS, add_propagators
from nucs.solvers.choice_points import backtrack, cp_init, cp_put
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_FCTS
from nucs.solvers.heuristics import (
    DOM_HEURISTIC_FCTS,
    DOM_HEURISTIC_MIN_VALUE,
    VAR_HEURISTIC_FCTS,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
)
from nucs.solvers.solver import Solver, decrease_max, get_solution, increase_min

logger = logging.getLogger(__name__)


class BacktrackSolver(Solver):
    """
    A solver relying on a backtracking mechanism.
    """

    def __init__(
        self,
        problem: Problem,
        consistency_alg_idx: int = CONSISTENCY_ALG_BC,
        var_heuristic_idx: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        dom_heuristic_idx: int = DOM_HEURISTIC_MIN_VALUE,
        stack_max_height: int = 128,
        log_level: str = LOG_LEVEL_INFO,
    ):
        """
        Inits the solver.
        :param problem: the problem to be solved
        :param consistency_alg_idx: the index of the consistency algorithm
        :param var_heuristic_idx: the index of the heuristic for selecting a variable/domain
        :param dom_heuristic_idx: the index of the heuristic for reducing a domain
        :param stack_max_height: the maximal choice point stack height
        :param log_level: the log level as a string
        """
        super().__init__(problem, log_level)
        logger.info(f"BacktrackSolver uses variable heuristic {var_heuristic_idx}")
        self.var_heuristic_idx = var_heuristic_idx
        logger.info(f"BacktrackSolver uses domain heuristic {dom_heuristic_idx}")
        self.dom_heuristic_idx = dom_heuristic_idx
        logger.info(f"BacktrackSolver uses consistency algorithm {consistency_alg_idx}")
        self.consistency_alg_idx = consistency_alg_idx
        self.triggered_propagators = np.ones(problem.propagator_nb, dtype=np.bool)
        logger.debug("Initializing choice points")
        self.shr_domains_stack = np.empty((stack_max_height, self.problem.variable_nb, 2), dtype=np.int32)
        self.not_entailed_propagators_stack = np.empty((stack_max_height, self.problem.propagator_nb), dtype=np.bool)
        self.dom_update_stack = np.empty((stack_max_height, 2), dtype=np.uint16)
        self.stacks_top = np.ones((1,), dtype=np.uint8)
        logger.info(f"Choice points stack has a maximal height of {stack_max_height}")
        cp_init(
            self.shr_domains_stack,
            self.not_entailed_propagators_stack,
            self.dom_update_stack,
            self.stacks_top,
            np.array(problem.shr_domains_lst),
        )
        logger.debug("Choice points initialized")
        logger.debug("Initializing statistics")
        self.statistics = np.array([0] * STATS_MAX, dtype=np.int64)
        logger.debug("Statistics initialized")
        logger.debug("BacktrackSolver initialized")

    def get_statistics(self) -> Dict[str, int]:
        return {
            STATS_LBL_ALG_BC_NB: int(self.statistics[STATS_IDX_ALG_BC_NB]),
            STATS_LBL_ALG_BC_WITH_SHAVING_NB: int(self.statistics[STATS_IDX_ALG_BC_WITH_SHAVING_NB]),
            STATS_LBL_ALG_SHAVING_NB: int(self.statistics[STATS_IDX_ALG_SHAVING_NB]),
            STATS_LBL_ALG_SHAVING_CHANGE_NB: int(self.statistics[STATS_IDX_ALG_SHAVING_CHANGE_NB]),
            STATS_LBL_ALG_SHAVING_NO_CHANGE_NB: int(self.statistics[STATS_IDX_ALG_SHAVING_NO_CHANGE_NB]),
            STATS_LBL_PROPAGATOR_ENTAILMENT_NB: int(self.statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB]),
            STATS_LBL_PROPAGATOR_FILTER_NB: int(self.statistics[STATS_IDX_PROPAGATOR_FILTER_NB]),
            STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB: int(self.statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB]),
            STATS_LBL_PROPAGATOR_INCONSISTENCY_NB: int(self.statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB]),
            STATS_LBL_SOLVER_BACKTRACK_NB: int(self.statistics[STATS_IDX_SOLVER_BACKTRACK_NB]),
            STATS_LBL_SOLVER_CHOICE_NB: int(self.statistics[STATS_IDX_SOLVER_CHOICE_NB]),
            STATS_LBL_SOLVER_CHOICE_DEPTH: int(self.statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]),
            STATS_LBL_SOLVER_SOLUTION_NB: int(self.statistics[STATS_IDX_SOLVER_SOLUTION_NB]),
        }

    def minimize(self, variable_idx: int) -> Optional[NDArray]:
        """
        Return the solution that minimizes a variable.
        :param variable_idx: the index of the variable to minimize
        :return: the optimal solution if it exists or None
        """
        logger.info(f"Minimizing variable {variable_idx}")
        return self.optimize(variable_idx, decrease_max)

    def maximize(self, variable_idx: int) -> Optional[NDArray]:
        """
        Return the solution that maximizes a variable.
        :param variable_idx: the index of the variable to maximize
        :return: the optimal solution if it exists or None
        """
        logger.info(f"Maximizing variable {variable_idx}")
        return self.optimize(variable_idx, increase_min)

    def optimize(self, variable_idx: int, update_domain_fct: Callable) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that optimizes a given variable.
        :param variable_idx: the index of the variable
        :param update_domain_fct: the function to update the domain of the variable
        :return: the solution if it exists or None
        """
        logger.debug(f"Optimizing variable {variable_idx}")
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs, consistency_alg_addrs = (
            get_function_addresses()
        )
        best_solution = None
        while (
            solution := solve_one(
                self.statistics,
                self.problem.algorithms,
                self.problem.var_bounds,
                self.problem.param_bounds,
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                self.problem.props_dom_indices,
                self.problem.props_dom_offsets,
                self.problem.props_parameters,
                self.problem.shr_domains_propagators,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
        ) is not None:
            logger.info(f"Found a local optimum: {solution[variable_idx]}")
            best_solution = solution
            reset(
                self.problem,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
            )
            update_domain_fct(
                self.shr_domains_stack,
                self.stacks_top,
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                variable_idx,
                best_solution[variable_idx],
            )
        return best_solution

    def solve(self) -> Iterator[NDArray]:
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        logger.info("Solving and iterating over the solutions")
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs, consistency_alg_addrs = (
            get_function_addresses()
        )
        while True:
            solution = solve_one(
                self.statistics,
                self.problem.algorithms,
                self.problem.var_bounds,
                self.problem.param_bounds,
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                self.problem.props_dom_indices,
                self.problem.props_dom_offsets,
                self.problem.props_parameters,
                self.problem.shr_domains_propagators,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            yield solution
            if not backtrack(
                self.statistics,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
                self.problem.shr_domains_propagators,
            ):
                break

    def minimize_and_queue(self, variable_idx: int, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solution that minimizes a variable.
        :param variable_idx: the index of the variable to minimize
        :param processor_idx: the index of the processor running the minimizer
        :param solution_queue: the solution queue
        """
        logger.info(f"Minimizing variable {variable_idx} and queuing solutions found")
        self.optimize_and_queue(variable_idx, decrease_max, processor_idx, solution_queue)

    def maximize_and_queue(self, variable_idx: int, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solution that maximizes a variable.
        :param variable_idx: the index of the variable to maximizer
        :param processor_idx: the index of the processor running the maximizer
        :param solution_queue: the solution queue
        """
        logger.info(f"Maximizing variable {variable_idx} and queuing solutions found")
        self.optimize_and_queue(variable_idx, increase_min, processor_idx, solution_queue)

    def optimize_and_queue(
        self, variable_idx: int, update_domain_fct: Callable, processor_idx: int, solution_queue: Queue
    ) -> None:
        """
        Enqueues the solution that optimizes a variable.
        :param variable_idx: the index of the variable
        :param update_domain_fct: the function to update the domain of the variable
        :param processor_idx: the index of the processor
        :param solution_queue: the solution queue
        """
        logger.debug(f"Optimizing variable {variable_idx} and queuing solutions found")
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs, consistency_alg_addrs = (
            get_function_addresses()
        )
        while True:
            solution = solve_one(
                self.statistics,
                self.problem.algorithms,
                self.problem.var_bounds,
                self.problem.param_bounds,
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                self.problem.props_dom_indices,
                self.problem.props_dom_offsets,
                self.problem.props_parameters,
                self.problem.shr_domains_propagators,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            logger.info(f"Found a local optimum: {solution[variable_idx]}")
            solution_queue.put((processor_idx, solution, self.statistics))
            reset(
                self.problem,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
            )
            update_domain_fct(
                self.shr_domains_stack,
                self.stacks_top,
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                variable_idx,
                solution[variable_idx],
            )
        solution_queue.put((processor_idx, None, self.statistics))

    def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solutions.
        :param processor_idx: the index of the processor
        :param solution_queue: the solution queue
        """
        logger.info("Solving and queuing solutions found")
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs, consistency_alg_addrs = (
            get_function_addresses()
        )
        while True:
            solution = solve_one(
                self.statistics,
                self.problem.algorithms,
                self.problem.var_bounds,
                self.problem.param_bounds,
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                self.problem.props_dom_indices,
                self.problem.props_dom_offsets,
                self.problem.props_parameters,
                self.problem.shr_domains_propagators,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            solution_queue.put((processor_idx, solution, self.statistics))
            if not backtrack(
                self.statistics,
                self.not_entailed_propagators_stack,
                self.dom_update_stack,
                self.stacks_top,
                self.triggered_propagators,
                self.problem.shr_domains_propagators,
            ):
                break
        solution_queue.put((processor_idx, None, self.statistics))


def reset(
    problem: Problem,
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    triggered_propagators: NDArray,
) -> None:
    """
    Resets the solver.
    :param problem: the problem
    :param shr_domains_stack: the stack of shared domains
    :param not_entailed_propagators_stack: the stack of not entailed propagators
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param triggered_propagators: the list of triggered propagators
    """
    cp_init(
        shr_domains_stack,
        not_entailed_propagators_stack,
        dom_update_stack,
        stacks_top,
        np.array(problem.shr_domains_lst),
    )
    triggered_propagators.fill(True)


@njit(cache=True)
def solve_one(
    statistics: NDArray,
    algorithms: NDArray,
    var_bounds: NDArray,
    param_bounds: NDArray,
    dom_indices_arr: NDArray,
    dom_offsets_arr: NDArray,
    props_dom_indices: NDArray,
    props_dom_offsets: NDArray,
    props_parameters: NDArray,
    shr_domains_propagators: NDArray,
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    triggered_propagators: NDArray,
    consistency_alg_idx: int,
    var_heuristic_idx: int,
    dom_heuristic_idx: int,
    compute_domains_addrs: NDArray,
    consistency_alg_addrs: NDArray,
    var_heuristic_addrs: NDArray,
    dom_heuristic_addrs: NDArray,
) -> Optional[NDArray]:
    """
    Find at most one solution.
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param var_bounds: the variable bounds indexed by propagators
    :param param_bounds: the parameters bounds indexed by propagators
    :param dom_indices_arr: the domain indices indexed by variables
    :param dom_offsets_arr: the domain offsets indexed by variables
    :param props_dom_indices: the domain indices indexed by propagator variables
    :param props_dom_offsets: the domain offsets indexed by propagator variables
    :param props_parameters: the parameters indexed by propagator variables
    :param shr_domains_propagators: a Numpy array of booleans indexed
    by shared domain indices, MIN/MAX and propagators; true means that the propagator has to be triggered when the MIN
    or MAX of the shared domain has changed
    :param shr_domains_stack: a stack of shared domains;
    the first level correspond to the current shared domains, the rest correspond to the choice points
    :param not_entailed_propagators_stack: a stack not entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param consistency_alg_idx: the index of the consistency algorithm
    :param var_heuristic_idx: the index of the variable heuristic
    :param dom_heuristic_idx: the index of the domain heuristic
    :param compute_domains_addrs: the addresses of the compute_domains functions
    :param consistency_alg_addrs: the addresses of the functions implementing the consistency algorithms
    :param var_heuristic_addrs: the addresses of the functions implementing the variable heuristics
    :param dom_heuristic_addrs: the addresses of the functions implementing the domain heuristics
    :return: the solution if it exists or None
    """
    if NUMBA_DISABLE_JIT:
        consistency_alg_fct = CONSISTENCY_ALG_FCTS[consistency_alg_idx]
        var_heuristic_fct = VAR_HEURISTIC_FCTS[var_heuristic_idx]
        dom_heuristic_fct = DOM_HEURISTIC_FCTS[dom_heuristic_idx]
    else:
        consistency_alg_fct = function_from_address(TYPE_CONSISTENCY_ALG, consistency_alg_addrs[consistency_alg_idx])
        var_heuristic_fct = function_from_address(TYPE_VAR_HEURISTIC, var_heuristic_addrs[var_heuristic_idx])
        dom_heuristic_fct = function_from_address(TYPE_DOM_HEURISTIC, dom_heuristic_addrs[dom_heuristic_idx])
    while True:
        status = consistency_alg_fct(
            statistics,
            algorithms,
            var_bounds,
            param_bounds,
            dom_indices_arr,
            dom_offsets_arr,
            props_dom_indices,
            props_dom_offsets,
            props_parameters,
            shr_domains_propagators,
            shr_domains_stack,
            not_entailed_propagators_stack,
            dom_update_stack,
            stacks_top,
            triggered_propagators,
            compute_domains_addrs,
        )
        if status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLVER_SOLUTION_NB] += 1
            return get_solution(shr_domains_stack, stacks_top, dom_indices_arr, dom_offsets_arr)
        elif status == PROBLEM_UNBOUND:
            dom_idx = var_heuristic_fct(shr_domains_stack, stacks_top)
            cp_put(shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx)
            event = dom_heuristic_fct(shr_domains_stack, dom_update_stack, stacks_top, dom_idx)
            add_propagators(
                triggered_propagators,
                not_entailed_propagators_stack,
                stacks_top,
                shr_domains_propagators,
                dom_idx,
                event,
            )
            statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
            if stacks_top[0] > statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]:
                statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = stacks_top[0]
        elif not backtrack(
            statistics,
            not_entailed_propagators_stack,
            dom_update_stack,
            stacks_top,
            triggered_propagators,
            shr_domains_propagators,
        ):
            return None


def get_function_addresses() -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Returns the addresses of the
    compute_domains, variable heuristics, domain heuristics and consistency algorithms functions.
    :return: a tuple of NDArrays
    """
    if NUMBA_DISABLE_JIT:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    return (
        np.array(build_function_address_list(COMPUTE_DOMAINS_FCTS, SIGNATURE_COMPUTE_DOMAINS)),
        np.array(build_function_address_list(VAR_HEURISTIC_FCTS, SIGNATURE_VAR_HEURISTIC)),
        np.array(build_function_address_list(DOM_HEURISTIC_FCTS, SIGNATURE_DOM_HEURISTIC)),
        np.array(build_function_address_list(CONSISTENCY_ALG_FCTS, SIGNATURE_CONSISTENCY_ALG)),
    )
