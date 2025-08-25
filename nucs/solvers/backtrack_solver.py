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
import logging
from multiprocessing import Queue
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    LOG_LEVEL_INFO,
    MAX,
    MIN,
    NUMBA_DISABLE_JIT,
    OPTIM_RESET,
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
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_LBL_ALG_BC_NB,
    STATS_LBL_ALG_BC_WITH_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_CHANGE_NB,
    STATS_LBL_ALG_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_ENTAILMENT_NB,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_INCONSISTENCY_NB,
    STATS_LBL_SOLUTION_NB,
    STATS_LBL_SOLVER_BACKTRACK_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
    STATS_LBL_SOLVER_CHOICE_NB,
    STATS_MAX,
    TYPE_CONSISTENCY_ALG,
    TYPE_DOM_HEURISTIC,
    TYPE_VAR_HEURISTIC,
)
from nucs.heaps import min_heap_init
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_FCTS,
    DOM_HEURISTIC_MIN_VALUE,
    VAR_HEURISTIC_FCTS,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
)
from nucs.numba_helper import build_function_address_list, function_from_address
from nucs.problems.problem import Problem
from nucs.propagators.propagators import COMPUTE_DOMAINS_FCTS, reset_triggered_propagators, update_propagators
from nucs.solvers.choice_points import backtrack, cp_init, fix_choice_points, fix_top_choice_point
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_FCTS
from nucs.solvers.queue_solver import QueueSolver
from nucs.solvers.solver import Solver, get_solution

logger = logging.getLogger(__name__)


class BacktrackSolver(Solver, QueueSolver):
    """
    A solver relying on a backtracking mechanism.
    """

    def __init__(
        self,
        problem: Problem,
        consistency_alg_idx: int = CONSISTENCY_ALG_BC,
        decision_variables: Optional[Iterable[int]] = None,
        var_heuristic_idx: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        var_heuristic_params: List[List[int]] = [[]],
        dom_heuristic_idx: int = DOM_HEURISTIC_MIN_VALUE,
        dom_heuristic_params: List[List[int]] = [[]],
        stks_max_height: int = 512,
        log_level: str = LOG_LEVEL_INFO,
    ):
        """
        Initializes the solver.
        :param problem: the problem to be solved
        :param consistency_alg_idx: the index of the consistency algorithm
        :param decision_variables: the variables on which decisions will be made
        :param var_heuristic_idx: the index of the heuristic for selecting a variable
        :param var_heuristic_params: a list of lists of parameters,
        usually parameters are costs and there is a list of value costs per variable
        :param dom_heuristic_idx: the index of the heuristic for reducing a domain
        :param dom_heuristic_params: a list of lists of parameters,
        usually parameters are costs and there is a list of value costs per variable
        :param stks_max_height: the maximal height of the choice point stacks
        :param log_level: the log level as a string
        """
        super().__init__(problem, log_level)
        decision_variables = range(problem.domain_nb) if decision_variables is None else decision_variables
        decision_variables = list(decision_variables)
        logger.info(f"BacktrackSolver uses decision domains {decision_variables}")
        self.decision_variables = np.array(decision_variables, dtype=np.uint32)
        logger.info(f"BacktrackSolver uses variable heuristic {var_heuristic_idx}")
        self.var_heuristic_idx = var_heuristic_idx
        self.var_heuristic_params = np.array(var_heuristic_params, dtype=np.int64)
        logger.info(f"BacktrackSolver uses domain heuristic {dom_heuristic_idx}")
        self.dom_heuristic_idx = dom_heuristic_idx
        self.dom_heuristic_params = np.array(dom_heuristic_params, dtype=np.int64)
        logger.info(f"BacktrackSolver uses consistency algorithm {consistency_alg_idx}")
        self.consistency_alg_idx = consistency_alg_idx
        self.triggered_propagators = min_heap_init(problem.propagator_nb)
        reset_triggered_propagators(self.triggered_propagators, self.problem.propagator_nb)
        logger.debug("Initializing choice points")
        self.domains_stk = np.empty((stks_max_height, self.problem.domain_nb, 2), dtype=np.int32)
        self.not_entailed_propagators_stk = np.empty((stks_max_height, self.problem.propagator_nb), dtype=np.bool)
        self.dom_update_stk = np.empty((stks_max_height, 2), dtype=np.uint32)
        self.stks_top = np.ones((1,), dtype=np.uint32)
        logger.info(f"The stacks of the choice points have a maximal height of {stks_max_height}")
        cp_init(
            self.domains_stk,
            self.not_entailed_propagators_stk,
            self.dom_update_stk,
            self.stks_top,
            np.array(problem.domains),
        )
        logger.debug("Choice points initialized")
        logger.debug("Initializing statistics")
        self.statistics = np.array([0] * STATS_MAX, dtype=np.int64)
        logger.debug("Statistics initialized")
        logger.debug("BacktrackSolver initialized")

    def get_statistics_as_array(self) -> NDArray:
        return self.statistics

    def get_statistics_as_dictionary(self) -> Dict[str, int]:
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
            STATS_LBL_SOLUTION_NB: int(self.statistics[STATS_IDX_SOLUTION_NB]),
        }

    def minimize(self, variable: int, mode: str = OPTIM_RESET) -> Optional[NDArray]:
        """
        Return the solution that minimizes a variable.
        :param variable: the variable to minimize
        :param mode: the optimization mode
        :return: the optimal solution if it exists or None
        """
        domain = self.domains_stk[self.stks_top[0], variable]
        logger.info(f"Minimizing (mode {mode}) variable {variable} (domain {domain}))")
        return self.optimize(variable, MAX, mode)

    def maximize(self, variable: int, mode: str = OPTIM_RESET) -> Optional[NDArray]:
        """
        Return the solution that maximizes a variable.
        :param variable: the variable to maximize
        :param mode: the optimization mode
        :return: the optimal solution if it exists or None
        """
        domain = self.domains_stk[self.stks_top[0], variable]
        logger.info(f"Maximizing (mode {mode}) variable {variable} (domain {domain}))")
        return self.optimize(variable, MIN, mode)

    def optimize(self, variable: int, bound: int, mode: str) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that optimizes a given variable.
        :param variable: the variable
        :param bound: the bound to optimize
        :param mode: the optimization mode
        :return: the solution if it exists or None
        """
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs, consistency_alg_addrs = (
            get_function_addresses()
        )
        best_solution = None
        while (
            solution := solve_one(
                self.problem.propagator_nb,
                self.statistics,
                self.problem.algorithms,
                self.problem.bounds,
                self.problem.props_variables,
                self.problem.props_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.not_entailed_propagators_stk,
                self.dom_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.decision_variables,
                self.var_heuristic_idx,
                self.var_heuristic_params,
                self.dom_heuristic_idx,
                self.dom_heuristic_params,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
        ) is not None:
            logger.info(f"Found a local optimum: {solution[variable]}")
            best_solution = solution
            if mode == OPTIM_RESET:
                logger.debug("Resetting solver")
                cp_init(
                    self.domains_stk,
                    self.not_entailed_propagators_stk,
                    self.dom_update_stk,
                    self.stks_top,
                    np.array(self.problem.domains),
                )
                if not fix_top_choice_point(
                    self.domains_stk,
                    self.stks_top,
                    variable,
                    best_solution[variable],
                    bound,
                ):
                    break
            else:
                logger.debug("Pruning choice points")
                if self.stks_top[0] == 0:
                    break
                self.stks_top[0] -= 1
                if not fix_choice_points(
                    self.domains_stk,
                    self.stks_top,
                    variable,
                    best_solution[variable],
                    bound,
                ):
                    break
            reset_triggered_propagators(self.triggered_propagators, self.problem.propagator_nb)
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
                self.problem.propagator_nb,
                self.statistics,
                self.problem.algorithms,
                self.problem.bounds,
                self.problem.props_variables,
                self.problem.props_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.not_entailed_propagators_stk,
                self.dom_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.decision_variables,
                self.var_heuristic_idx,
                self.var_heuristic_params,
                self.dom_heuristic_idx,
                self.dom_heuristic_params,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            logger.debug("Found a solution")
            yield solution
            if not backtrack(
                self.problem.propagator_nb,
                self.statistics,
                self.not_entailed_propagators_stk,
                self.dom_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.problem.triggers,
            ):
                break

    def minimize_and_queue(self, variable: int, processor_idx: int, solution_queue: Queue, mode: str) -> None:
        """
        Enqueues the solution that minimizes a variable.
        :param variable: the variable to minimize
        :param processor_idx: the index of the processor running the minimizer
        :param solution_queue: the solution queue
        :param mode: the optimization mode
        """
        domain = self.domains_stk[self.stks_top[0], variable]
        logger.info(f"Minimizing (mode {mode}) variable {variable} (domain {domain})) and queuing solutions")
        self.optimize_and_queue(variable, MAX, processor_idx, solution_queue, mode)

    def maximize_and_queue(self, variable: int, processor_idx: int, solution_queue: Queue, mode: str) -> None:
        """
        Enqueues the solution that maximizes a variable.
        :param variable: the variable to maximizer
        :param processor_idx: the index of the processor running the maximizer
        :param solution_queue: the solution queue
        :param mode: the optimization mode
        """
        domain = self.domains_stk[self.stks_top[0], variable]
        logger.info(f"Minimizing (mode {mode}) variable {variable} (domain {domain})) and queuing solutions")
        self.optimize_and_queue(variable, MIN, processor_idx, solution_queue, mode)

    def optimize_and_queue(
        self, variable: int, bound: int, processor_idx: int, solution_queue: Queue, mode: str
    ) -> None:
        """
        Enqueues the solution that optimizes a variable.
        :param variable: the variable
        :param bound: the bound to optimize
        :param processor_idx: the index of the processor
        :param solution_queue: the solution queue
        :param mode: the optimization mode
        """
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs, consistency_alg_addrs = (
            get_function_addresses()
        )
        while True:
            solution = solve_one(
                self.problem.propagator_nb,
                self.statistics,
                self.problem.algorithms,
                self.problem.bounds,
                self.problem.props_variables,
                self.problem.props_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.not_entailed_propagators_stk,
                self.dom_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.decision_variables,
                self.var_heuristic_idx,
                self.var_heuristic_params,
                self.dom_heuristic_idx,
                self.dom_heuristic_params,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            logger.info(f"Found a local optimum: {solution[variable]}")
            logger.info(self.stks_top[0])
            solution_queue.put((processor_idx, solution, self.statistics))
            if mode == OPTIM_RESET:
                logger.debug("Resetting solver")
                cp_init(
                    self.domains_stk,
                    self.not_entailed_propagators_stk,
                    self.dom_update_stk,
                    self.stks_top,
                    np.array(self.problem.domains),
                )
                if not fix_top_choice_point(
                    self.domains_stk,
                    self.stks_top,
                    variable,
                    solution[variable],
                    bound,
                ):
                    break
            else:
                logger.debug("Pruning choice points")
                if self.stks_top[0] == 0:
                    break
                self.stks_top[0] -= 1
                if not fix_choice_points(
                    self.domains_stk,
                    self.stks_top,
                    variable,
                    solution[variable],
                    bound,
                ):
                    break
            reset_triggered_propagators(self.triggered_propagators, self.problem.propagator_nb)
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
                self.problem.propagator_nb,
                self.statistics,
                self.problem.algorithms,
                self.problem.bounds,
                self.problem.props_variables,
                self.problem.props_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.not_entailed_propagators_stk,
                self.dom_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_idx,
                self.decision_variables,
                self.var_heuristic_idx,
                self.var_heuristic_params,
                self.dom_heuristic_idx,
                self.dom_heuristic_params,
                compute_domains_addrs,
                consistency_alg_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            solution_queue.put((processor_idx, solution, self.statistics))
            if not backtrack(
                self.problem.propagator_nb,
                self.statistics,
                self.not_entailed_propagators_stk,
                self.dom_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.problem.triggers,
            ):
                break
        solution_queue.put((processor_idx, None, self.statistics))


@njit(cache=True)
def solve_one(
    propagator_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    bounds: NDArray,
    props_variables: NDArray,
    props_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    not_entailed_propagators_stk: NDArray,
    dom_update_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    consistency_alg_idx: int,
    decision_variables: NDArray,
    var_heuristic_idx: int,
    var_heuristic_params: NDArray,
    dom_heuristic_idx: int,
    dom_heuristic_params: NDArray,
    compute_domains_addrs: NDArray,
    consistency_alg_addrs: NDArray,
    var_heuristic_addrs: NDArray,
    dom_heuristic_addrs: NDArray,
) -> Optional[NDArray]:
    """
    Find at most one solution.
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param bounds: the bounds indexed by propagators
    :param props_variables: the variables by propagators
    :param props_parameters: the parameters by propagators
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :param domains_stk: a stack of domains;
    the first level correspond to the current domains, the rest correspond to the choice points
    :param not_entailed_propagators_stk: a stack not entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param dom_update_stk: the stack of domain updates
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param consistency_alg_idx: the index of the consistency algorithm
    :param decision_variables: the variables on which decisions will be made
    :param var_heuristic_idx: the index of the variable heuristic
    :param var_heuristic_params: a list of lists of parameters,
    usually parameters are costs and there is a list of value costs per variable
    :param dom_heuristic_idx: the index of the domain heuristic
    :param dom_heuristic_params: a list of lists of parameters,
    usually parameters are costs and there is a list of value costs per variable
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
            propagator_nb,
            statistics,
            algorithms,
            bounds,
            props_variables,
            props_parameters,
            triggers,
            domains_stk,
            not_entailed_propagators_stk,
            dom_update_stk,
            stks_top,
            triggered_propagators,
            compute_domains_addrs,
            decision_variables,
        )
        if status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLUTION_NB] += 1
            return get_solution(domains_stk, stks_top)
        elif status == PROBLEM_UNBOUND:
            var = var_heuristic_fct(decision_variables, domains_stk, stks_top, var_heuristic_params)
            events = dom_heuristic_fct(
                domains_stk,
                not_entailed_propagators_stk,
                dom_update_stk,
                stks_top,
                var,
                dom_heuristic_params,
            )
            update_propagators(
                propagator_nb, triggered_propagators, not_entailed_propagators_stk[stks_top[0]], triggers, events, var
            )
            statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
            if stks_top[0] > statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]:
                statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = stks_top[0]
        elif not backtrack(
            propagator_nb,
            statistics,
            not_entailed_propagators_stk,
            dom_update_stk,
            stks_top,
            triggered_propagators,
            triggers,
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
