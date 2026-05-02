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
)
from nucs.heaps import min_heap_init
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
    build_var_heuristic_fcts, address_from_function,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    COMPUTE_DOMAINS_FCTS,
    reset_triggered_propagators,
    update_propagators,
    get_algorithm_nb,
)
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
        consistency_alg: int = CONSISTENCY_ALG_BC,
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
        :param consistency_alg: the index of the consistency algorithm
        :param decision_variables: the variables on which decisions will be made
        :param var_heuristic: the index of the heuristic for selecting a variable
        :param var_heuristic_params: a list of lists of parameters,
        usually parameters are costs and there is a list of value costs per variable
        :param dom_heuristic: the index of the heuristic for reducing a domain
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
        logger.info(f"BacktrackSolver uses variable heuristic {var_heuristic}")
        self.var_heuristic_params = np.array(var_heuristic_params, dtype=np.int64)
        logger.info(f"BacktrackSolver uses domain heuristic {dom_heuristic}")
        self.dom_heuristic_params = np.array(dom_heuristic_params, dtype=np.int64)
        logger.info(f"BacktrackSolver uses consistency algorithm {consistency_alg}")
        self.triggered_propagators = min_heap_init(problem.propagator_nb)
        reset_triggered_propagators(self.triggered_propagators, self.problem.complexities)
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
            self.consistency_alg_fcts = [CONSISTENCY_ALG_FCTS[consistency_alg]]
            self.var_heuristic_fcts = [VAR_HEURISTIC_FCTS[var_heuristic]]
            self.dom_heuristic_fcts = [DOM_HEURISTIC_FCTS[dom_heuristic]]
        else:
            compute_domains_addrs = addresses_from_functions(COMPUTE_DOMAINS_FCTS, SIGNATURE_COMPUTE_DOMAINS)
            self.compute_domains_fcts = build_compute_domains_fcts(compute_domains_addrs)
            consistency_alg_addr = address_from_function(CONSISTENCY_ALG_FCTS[consistency_alg],
                                                         SIGNATURE_CONSISTENCY_ALG)
            self.consistency_alg_fcts = build_consistency_alg_fcts(consistency_alg_addr)
            var_heuristic_addr = address_from_function(VAR_HEURISTIC_FCTS[var_heuristic], SIGNATURE_VAR_HEURISTIC)
            self.var_heuristic_fcts = build_var_heuristic_fcts(var_heuristic_addr)
            dom_heuristic_addr = address_from_function(DOM_HEURISTIC_FCTS[dom_heuristic], SIGNATURE_DOM_HEURISTIC)
            self.dom_heuristic_fcts = build_dom_heuristic_fcts(dom_heuristic_addr)
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
        best_solution = None
        while (
            solution := solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.complexities,
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
            reset_triggered_propagators(self.triggered_propagators, self.problem.complexities)
        return best_solution

    def solve(self) -> Iterator[NDArray]:
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        logger.info("Solving and iterating over the solutions")
        while True:
            solution = solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.complexities,
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
                self.problem.complexities,
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
        :param variable: the variable to maximize
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
        while True:
            solution = solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.complexities,
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
            reset_triggered_propagators(self.triggered_propagators, self.problem.complexities)
        solution_queue.put((processor_idx, None, self.statistics))

    def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solutions.
        :param processor_idx: the index of the processor
        :param solution_queue: the solution queue
        """
        logger.info("Solving and queuing solutions found")
        while True:
            solution = solve_one(
                get_algorithm_nb(),
                self.statistics,
                self.problem.algorithms,
                self.problem.complexities,
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
                self.problem.complexities,
            ):
                break
        solution_queue.put((processor_idx, None, self.statistics))


@njit(cache=True, fastmath=True)
def solve_one(
    algorithm_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    complexities: NDArray,
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
) -> Optional[NDArray]:
    """
    Find at most one solution.
    :param statistics: a Numpy array of statistics
    :param algorithms: the algorithms indexed by propagators
    :param bounds: the bounds indexed by propagators
    :param propagator_variables: the variables by propagators
    :param propagator_parameters: the parameters by propagators
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :param domains_stk: a stack of domains;
    the first level correspond to the current domains, the rest correspond to the choice points
    :param entailed_propagators_stk: a stack of entailed propagators;
    the first level correspond to the propagators currently not entailed, the rest correspond to the choice points
    :param domain_update_stk: the stack of domain updates
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :param stks_top: the index of the top of the stacks as a Numpy array
    :param triggered_propagators: the Numpy array of triggered propagators
    :param consistency_alg_fcts: a 1-element list holding the consistency algorithm function
    :param decision_variables: the variables on which decisions will be made
    :param var_heuristic_fcts: a 1-element list holding the variable heuristic function
    :param var_heuristic_params: a list of lists of parameters,
    usually parameters are costs and there is a list of value costs per variable
    :param dom_heuristic_fcts: a 1-element list holding the domain heuristic function
    :param dom_heuristic_params: a list of lists of parameters,
    usually parameters are costs and there is a list of value costs per variable
    :param compute_domains_fcts: the typed list of compute_domains functions, built once at solver init
    :return: the solution if it exists or None
    """
    consistency_alg_fct = consistency_alg_fcts[0]
    var_heuristic_fct = var_heuristic_fcts[0]
    dom_heuristic_fct = dom_heuristic_fcts[0]
    while True:
        status = consistency_alg_fct(
            algorithm_nb,
            statistics,
            algorithms,
            complexities,
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
                triggered_propagators, entailed_propagators_stk[top], triggers[variable, events], complexities
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
            complexities,
        ):
            return None
