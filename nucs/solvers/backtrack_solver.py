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
from multiprocessing import Queue
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import buckets_init, buckets_empty, buckets_add
from nucs.constants import (
    LOG_LEVEL_INFO,
    MAX,
    MIN,
    NUMBA_DISABLE_JIT,
    OPTIM_RESET,
    PROBLEM_BOUND,
    PROBLEM_UNBOUND,
    SIGN_COMPUTE_DOMAINS,
    SIGN_CONSISTENCY_ALG,
    SIGN_DOM_HEURISTIC,
    SIGN_VAR_HEURISTIC,
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
)
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_FCTS,
    DOM_HEURISTIC_MIN_VALUE,
    VAR_HEURISTIC_FCTS,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
)
from nucs.numba_helper import (
    addresses_from_functions,
    build_compute_domains_fcts,
    build_consistency_alg_fcts,
    build_dom_heuristic_fcts,
    build_var_heuristic_fcts,
    address_from_function,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    COMPUTE_DOMAINS_FCTS,
    update_propagators,
    get_algorithm_nb,
)
from nucs.solvers.bound_consistency_algorithm import get_domain_buffer
from nucs.solvers.choice_points import backtrack, cp_init, fix_choice_points, fix_choice_point
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
        consistency_algorithm: int = CONSISTENCY_ALG_BC,
        decision_variables: Optional[Iterable[int]] = None,
        var_heuristic: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        var_heuristic_params: List[List[int]] = [[]],
        dom_heuristic: int = DOM_HEURISTIC_MIN_VALUE,
        dom_heuristic_params: List[List[int]] = [[]],
        stks_max_height: int = 512,
        log_level: str = LOG_LEVEL_INFO,
    ):
        """
        Initializes the solver.

        :param problem: the problem to be solved
        :type problem: Problem
        :param consistency_algorithm: the consistency algorithm, defaults to bound consistency
        :type consistency_algorithm: int
        :param decision_variables: the variables on which decisions will be made, defaults to None
        :type decision_variables: Optional[Iterable[int]]
        :param var_heuristic: the heuristic for selecting a variable,
                              defaults to the first non instantiated
        :type var_heuristic: int
        :param var_heuristic_params: a list of lists of parameters,
                                     usually parameters are costs and there is a list of value costs per variable
        :type var_heuristic_params: List[List[int]]
        :param dom_heuristic: the heuristic for reducing a domain,
                              defaults to instantiating the domain to its first value
        :type dom_heuristic: int
        :param dom_heuristic_params: a list of lists of parameters,
                                     usually parameters are costs and there is a list of value costs per variable
        :type dom_heuristic_params: List[List[int]]
        :param stks_max_height: the maximal height of the choice point stacks,
                                defaults to 512
        :type stks_max_height: int
        :param log_level: the log level,
                          defaults to INFO
        :type log_level: str
        """
        super().__init__(problem, log_level)
        decision_variables = list(range(problem.domain_nb)) if decision_variables is None else list(decision_variables)
        logger.info(f"BacktrackSolver uses decision domains {decision_variables}")
        self.decision_variables = np.array(decision_variables, dtype=np.uint32)
        logger.info(f"BacktrackSolver uses variable heuristic {var_heuristic}")
        self.var_heuristic_params = np.array(var_heuristic_params, dtype=np.int64)
        logger.info(f"BacktrackSolver uses domain heuristic {dom_heuristic}")
        self.dom_heuristic_params = np.array(dom_heuristic_params, dtype=np.int64)
        logger.info(f"BacktrackSolver uses consistency algorithm {consistency_algorithm}")
        self.triggered_propagators = buckets_init(problem.propagator_nb)
        self.domain_buffer = get_domain_buffer(problem.bounds)
        logger.debug("Initializing choice points")
        self.domains_stk = np.empty((stks_max_height, self.problem.domain_nb, 2), dtype=np.int32)
        self.entailed_propagators_stk = np.empty((stks_max_height, self.problem.propagator_nb), dtype=np.bool)
        self.domain_update_stk = np.empty((stks_max_height, 2), dtype=np.uint32)
        self.unbound_variable_nb_stk = np.empty(stks_max_height, dtype=np.uint32)
        self.stks_top = np.ones((1,), dtype=np.uint32)
        logger.info(f"The stacks of the choice points have a maximal height of {stks_max_height}")
        self.initial_domains = np.array(problem.domains)
        cp_init(
            self.domains_stk,
            self.entailed_propagators_stk,
            self.domain_update_stk,
            self.unbound_variable_nb_stk,
            self.stks_top,
            self.initial_domains,
            problem.unbound_variable_nb,
        )
        logger.debug("Choice points initialized")
        logger.debug("Initializing statistics")
        self.statistics = np.zeros(STATS_MAX, dtype=np.int64)
        logger.debug("Statistics initialized")
        if NUMBA_DISABLE_JIT:
            self.compute_domains_fcts = COMPUTE_DOMAINS_FCTS
            self.consistency_alg_fcts = [CONSISTENCY_ALG_FCTS[consistency_algorithm]]
            self.var_heuristic_fcts = [VAR_HEURISTIC_FCTS[var_heuristic]]
            self.dom_heuristic_fcts = [DOM_HEURISTIC_FCTS[dom_heuristic]]
        else:
            compute_domains_addrs = addresses_from_functions(COMPUTE_DOMAINS_FCTS, SIGN_COMPUTE_DOMAINS)
            self.compute_domains_fcts = build_compute_domains_fcts(compute_domains_addrs)
            consistency_alg_addr = address_from_function(
                CONSISTENCY_ALG_FCTS[consistency_algorithm], SIGN_CONSISTENCY_ALG
            )
            self.consistency_alg_fcts = build_consistency_alg_fcts(consistency_alg_addr)
            var_heuristic_addr = address_from_function(VAR_HEURISTIC_FCTS[var_heuristic], SIGN_VAR_HEURISTIC)
            self.var_heuristic_fcts = build_var_heuristic_fcts(var_heuristic_addr)
            dom_heuristic_addr = address_from_function(DOM_HEURISTIC_FCTS[dom_heuristic], SIGN_DOM_HEURISTIC)
            self.dom_heuristic_fcts = build_dom_heuristic_fcts(dom_heuristic_addr)
        logger.debug("BacktrackSolver initialized")

    def get_statistics_as_array(self) -> NDArray:
        """
        Returns the statistics.
        """
        return self.statistics

    def get_statistics_as_dictionary(self) -> Dict[str, int]:
        """
        Returns the statistics.
        """
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
        :type variable: int
        :param mode: the optimization mode (RESET or PRUNE), defaults to RESET
        :type mode: str

        :return: the optimal solution if it exists or None
        :rtype: Optional[NDArray]
        """
        domain = self.domains_stk[self.stks_top[0], variable]
        logger.info(f"Minimizing (mode {mode}) variable {variable} (domain {domain}))")
        return self.optimize(variable, MAX, mode)

    def maximize(self, variable: int, mode: str = OPTIM_RESET) -> Optional[NDArray]:
        """
        Return the solution that maximizes a variable.

        :param variable: the variable to maximize
        :type variable: int
        :param mode: the optimization mode (RESET or PRUNE), defaults to RESET
        :type mode: str

        :return: the optimal solution if it exists or None
        :rtype: Optional[NDArray]
        """
        domain = self.domains_stk[self.stks_top[0], variable]
        logger.info(f"Maximizing (mode {mode}) variable {variable} (domain {domain}))")
        return self.optimize(variable, MIN, mode)

    def optimize(self, variable: int, bound: int, mode: str) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that optimizes a given variable.

        :param variable: the variable
        :type variable: int
        :param bound: the bound to optimize
        :type bound: int
        :param mode: the optimization mode
        :type mode: str

        :return: the solution if it exists or None
        :rtype: Optional[NDArray]
        """
        best_solution = None
        while (
            solution := solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.priorities,
                self.problem.bounds,
                self.problem.propagator_variables,
                self.problem.propagator_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.entailed_propagators_stk,
                self.domain_update_stk,
                self.unbound_variable_nb_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_fcts,
                self.decision_variables,
                self.var_heuristic_fcts,
                self.var_heuristic_params,
                self.dom_heuristic_fcts,
                self.dom_heuristic_params,
                self.compute_domains_fcts,
                self.domain_buffer,
            )
        ) is not None:
            logger.info(f"Found a local optimum: {solution[variable]}")
            best_solution = solution
            if mode == OPTIM_RESET:
                logger.debug("Resetting solver")
                cp_init(
                    self.domains_stk,
                    self.entailed_propagators_stk,
                    self.domain_update_stk,
                    self.unbound_variable_nb_stk,
                    self.stks_top,
                    self.initial_domains,
                    self.problem.unbound_variable_nb,
                )
                if not fix_choice_point(
                    self.domains_stk,
                    self.unbound_variable_nb_stk,
                    variable,
                    best_solution[variable],
                    bound,
                ):
                    break
            else:
                logger.debug("Pruning choice points")
                if not fix_choice_points(
                    self.domains_stk,
                    self.unbound_variable_nb_stk,
                    self.stks_top,
                    variable,
                    best_solution[variable],
                    bound,
                ):
                    break
        return best_solution

    def solve(self) -> Iterator[NDArray]:
        """
        Returns an iterator over the solutions.

        :return: an iterator
        :rtype: Iterator[NDArray]
        """
        logger.info("Solving and iterating over the solutions")
        while True:
            solution = solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.priorities,
                self.problem.bounds,
                self.problem.propagator_variables,
                self.problem.propagator_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.entailed_propagators_stk,
                self.domain_update_stk,
                self.unbound_variable_nb_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_fcts,
                self.decision_variables,
                self.var_heuristic_fcts,
                self.var_heuristic_params,
                self.dom_heuristic_fcts,
                self.dom_heuristic_params,
                self.compute_domains_fcts,
                self.domain_buffer,
            )
            if solution is None:
                break
            logger.debug("Found a solution")
            yield solution
            if not backtrack(
                self.statistics,
                self.entailed_propagators_stk,
                self.domain_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.problem.triggers,
                self.problem.priorities,
            ):
                break

    def minimize_and_queue(self, variable: int, processor_idx: int, solution_queue: Queue, mode: str) -> None:
        """
        Enqueues the solution that minimizes a variable.

        :param variable: the variable to minimize
        :type variable: int
        :param processor_idx: the index of the processor running the minimizer
        :type processor_idx: int
        :param solution_queue: the solution queue
        :type solution_queue: Queue
        :param mode: the optimization mode
        :type mode: str
        """
        domain = self.domains_stk[self.stks_top[0], variable]
        logger.info(f"Minimizing (mode {mode}) variable {variable} (domain {domain})) and queuing solutions")
        self.optimize_and_queue(variable, MAX, processor_idx, solution_queue, mode)

    def maximize_and_queue(self, variable: int, processor_idx: int, solution_queue: Queue, mode: str) -> None:
        """
        Enqueues the solution that maximizes a variable.

        :param variable: the variable to maximize
        :type variable: int
        :param processor_idx: the index of the processor running the maximizer
        :type processor_idx: int
        :param solution_queue: the solution queue
        :type solution_queue: Queue
        :param mode: the optimization mode
        :type mode: str
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
        :type variable: int
        :param bound: the bound to optimize
        :type bound: int
        :param processor_idx: the index of the processor
        :type processor_idx: int
        :param solution_queue: the solution queue
        :type solution_queue: Queue
        :param mode: the optimization mode
        :type mode: str
        """
        while True:
            solution = solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.priorities,
                self.problem.bounds,
                self.problem.propagator_variables,
                self.problem.propagator_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.entailed_propagators_stk,
                self.domain_update_stk,
                self.unbound_variable_nb_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_fcts,
                self.decision_variables,
                self.var_heuristic_fcts,
                self.var_heuristic_params,
                self.dom_heuristic_fcts,
                self.dom_heuristic_params,
                self.compute_domains_fcts,
                self.domain_buffer,
            )
            if solution is None:
                break
            logger.info(f"Found a local optimum: {solution[variable]}")
            solution_queue.put((processor_idx, solution, self.statistics))
            if mode == OPTIM_RESET:
                logger.debug("Resetting solver")
                cp_init(
                    self.domains_stk,
                    self.entailed_propagators_stk,
                    self.domain_update_stk,
                    self.unbound_variable_nb_stk,
                    self.stks_top,
                    self.initial_domains,
                    self.problem.unbound_variable_nb,
                )
                if not fix_choice_point(
                    self.domains_stk,
                    self.unbound_variable_nb_stk,
                    variable,
                    solution[variable],
                    bound,
                ):
                    break
            else:
                logger.debug("Pruning choice points")
                if not fix_choice_points(
                    self.domains_stk,
                    self.unbound_variable_nb_stk,
                    self.stks_top,
                    variable,
                    solution[variable],
                    bound,
                ):
                    break
        solution_queue.put((processor_idx, None, self.statistics))

    def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solutions.

        :param processor_idx: the index of the processor
        :type processor_idx: int
        :param solution_queue: the solution queue
        :type solution_queue: Queue
        """
        logger.info("Solving and queuing solutions found")
        while True:
            solution = solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.priorities,
                self.problem.bounds,
                self.problem.propagator_variables,
                self.problem.propagator_parameters,
                self.problem.triggers,
                self.domains_stk,
                self.entailed_propagators_stk,
                self.domain_update_stk,
                self.unbound_variable_nb_stk,
                self.stks_top,
                self.triggered_propagators,
                self.consistency_alg_fcts,
                self.decision_variables,
                self.var_heuristic_fcts,
                self.var_heuristic_params,
                self.dom_heuristic_fcts,
                self.dom_heuristic_params,
                self.compute_domains_fcts,
                self.domain_buffer,
            )
            if solution is None:
                break
            solution_queue.put((processor_idx, solution, self.statistics))
            if not backtrack(
                self.statistics,
                self.entailed_propagators_stk,
                self.domain_update_stk,
                self.stks_top,
                self.triggered_propagators,
                self.problem.triggers,
                self.problem.priorities,
            ):
                break
        solution_queue.put((processor_idx, None, self.statistics))


@njit(cache=True, fastmath=True)
def solve_one(
    algorithm_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    priorities: NDArray,
    bounds: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    domains_stk: NDArray,
    entailed_propagators_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    consistency_alg_fcts: Any,
    decision_variables: NDArray,
    var_heuristic_fcts: Any,
    var_heuristic_params: NDArray,
    dom_heuristic_fcts: Any,
    dom_heuristic_params: NDArray,
    compute_domains_fcts: Any,
    domain_buffer: NDArray,
) -> Optional[NDArray]:
    """
    Find at most one solution.

    :param statistics: a Numpy array of statistics
    :type statistics: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param bounds: the bounds indexed by propagators
    :type bounds: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param domains_stk: a stack of domains,
                        the first level correspond to the current domains, the rest correspond to the choice points
    :type domains_stk: NDArray
    :param entailed_propagators_stk: a stack of entailed propagators,
                                     the first level correspond to the propagators currently not entailed,
                                     the rest correspond to the choice points
    :type entailed_propagators_stk: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param consistency_alg_fcts: a 1-element list holding the consistency algorithm function
    :type consistency_alg_fcts: Any
    :param decision_variables: the variables on which decisions will be made
    :type decision_variables: NDArray
    :param var_heuristic_fcts: a 1-element list holding the variable heuristic function
    :type var_heuristic_fcts: Any
    :param var_heuristic_params: a list of lists of parameters,
    usually parameters are costs and there is a list of value costs per variable
    :type var_heuristic_params: NDArray
    :param dom_heuristic_fcts: a 1-element list holding the domain heuristic function
    :type dom_heuristic_fcts: Any
    :param dom_heuristic_params: a list of lists of parameters,
                                 usually parameters are costs and there is a list of value costs per variable
    :type dom_heuristic_params: NDArray
    :param compute_domains_fcts: the typed list of compute_domains functions, built once at solver init
    :type compute_domains_fcts: Any

    :return: the solution if it exists or None
    :rtype: Optional[NDArray]
    """
    consistency_alg_fct = consistency_alg_fcts[0]
    var_heuristic_fct = var_heuristic_fcts[0]
    dom_heuristic_fct = dom_heuristic_fcts[0]
    buckets_empty(triggered_propagators, priorities)
    for prop_idx in range(len(priorities)):
        buckets_add(triggered_propagators, prop_idx, priorities)
    while True:
        status = consistency_alg_fct(
            algorithm_nb,
            statistics,
            algorithms,
            priorities,
            bounds,
            propagator_variables,
            propagator_parameters,
            triggers,
            domains_stk,
            entailed_propagators_stk,
            domain_update_stk,
            unbound_variable_nb_stk,
            stks_top,
            triggered_propagators,
            compute_domains_fcts,
            decision_variables,
            domain_buffer,
        )
        top = stks_top[0]
        if status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLUTION_NB] += 1
            return get_solution(domains_stk, top)
        elif status == PROBLEM_UNBOUND:
            variable = var_heuristic_fct(decision_variables, domains_stk, top, var_heuristic_params)
            events = dom_heuristic_fct(
                domains_stk,
                entailed_propagators_stk,
                domain_update_stk,
                unbound_variable_nb_stk,
                stks_top,
                variable,
                dom_heuristic_params,
            )
            top = stks_top[0]
            update_propagators(
                triggered_propagators, entailed_propagators_stk[top], triggers[variable, events], priorities
            )
            statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
            if top > statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]:
                statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = top
        elif not backtrack(
            statistics,
            entailed_propagators_stk,
            domain_update_stk,
            stks_top,
            triggered_propagators,
            triggers,
            priorities,
        ):
            return None
