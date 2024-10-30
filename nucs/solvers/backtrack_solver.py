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
from multiprocessing import Queue
from typing import Callable, Iterator, Optional

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    NUMBA_DISABLE_JIT,
    PROBLEM_INCONSISTENT,
    PROBLEM_SOLVED,
    PROBLEM_TO_FILTER,
    STACK_MAX_HEIGHT,
    TYPE_DOM_HEURISTIC,
    TYPE_VAR_HEURISTIC,
)
from nucs.numba_helper import function_from_address, get_function_addresses
from nucs.problems.problem import Problem
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, consistency_algorithm
from nucs.solvers.heuristics import (
    DOM_HEURISTIC_FCTS,
    DOM_HEURISTIC_MIN_VALUE,
    VAR_HEURISTIC_FCTS,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
)
from nucs.solvers.solver import Solver, decrease_max, get_solution, increase_min
from nucs.statistics import (
    STATS_IDX_OPTIMIZER_SOLUTION_NB,
    STATS_IDX_PROBLEM_PROPAGATOR_NB,
    STATS_IDX_PROBLEM_VARIABLE_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_SOLUTION_NB,
    init_statistics,
)


class BacktrackSolver(Solver):
    """
    A solver relying on a backtracking mechanism.
    """

    def __init__(
        self,
        problem: Problem,
        consistency_algorithm_idx: int = CONSISTENCY_ALG_BC,
        var_heuristic_idx: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        dom_heuristic_idx: int = DOM_HEURISTIC_MIN_VALUE,
    ):
        """
        Inits the solver.
        :param problem: the problem
        :param var_heuristic_idx: a heuristic for selecting a variable/domain
        :param dom_heuristic_idx: a heuristic for reducing a domain
        """
        self.var_heuristic_idx = var_heuristic_idx
        self.dom_heuristic_idx = dom_heuristic_idx
        self.consistency_algorithm_idx = consistency_algorithm_idx
        self.triggered_propagators = np.ones(problem.propagator_nb, dtype=np.bool)
        problem.init()
        self.problem = problem
        self.shr_domains_stack = np.empty((STACK_MAX_HEIGHT, self.problem.variable_nb, 2), dtype=np.int32)
        self.shr_domains_stack[0] = problem.shr_domains_lst
        self.not_entailed_propagators_stack = np.empty((STACK_MAX_HEIGHT, self.problem.propagator_nb), dtype=np.bool)
        self.not_entailed_propagators_stack[0] = True
        self.stacks_height = np.ones((1,), dtype=np.uint8)
        self.statistics = init_statistics()
        self.statistics[STATS_IDX_PROBLEM_PROPAGATOR_NB] = self.problem.propagator_nb
        self.statistics[STATS_IDX_PROBLEM_VARIABLE_NB] = self.problem.variable_nb

    def minimize(self, variable_idx: int) -> Optional[NDArray]:
        """
        Return the solution that minimizes a variable.
        :param variable_idx: the index of the variable to minimize
        :return: the optimal solution if it exists or None
        """
        return self.optimize(variable_idx, decrease_max)

    def maximize(self, variable_idx: int) -> Optional[NDArray]:
        """
        Return the solution that maximizes a variable.
        :param variable_idx: the index of the variable to maximize
        :return: the optimal solution if it exists or None
        """
        return self.optimize(variable_idx, increase_min)

    def optimize(self, variable_idx: int, update_domain: Callable) -> Optional[NDArray]:
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs = get_function_addresses()
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
                self.stacks_height,
                self.triggered_propagators,
                self.consistency_algorithm_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
        ) is not None:
            best_solution = solution
            self.statistics[STATS_IDX_OPTIMIZER_SOLUTION_NB] += 1
            reset(
                self.problem,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.stacks_height,
                self.triggered_propagators,
            )
            update_domain(
                self.shr_domains_stack[0],
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
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs = get_function_addresses()
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
                self.stacks_height,
                self.triggered_propagators,
                self.consistency_algorithm_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            yield solution
            if not backtrack(
                self.statistics,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.stacks_height,
                self.triggered_propagators,
            ):
                break

    def minimize_and_queue(self, variable_idx: int, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solution that minimizes a variable.
        :param variable_idx: the index of the variable to minimize
        :param processor_idx: the index of the processor running the minimizer
        :param solution_queue: the solution queue
        """
        self.optimize_and_queue(variable_idx, decrease_max, processor_idx, solution_queue)

    def maximize_and_queue(self, variable_idx: int, processor_idx: int, solution_queue: Queue) -> None:
        """
        Enqueues the solution that maximizes a variable.
        :param variable_idx: the index of the variable to maximizer
        :param processor_idx: the index of the processor running the maximizer
        :param solution_queue: the solution queue
        """
        self.optimize_and_queue(variable_idx, increase_min, processor_idx, solution_queue)

    def optimize_and_queue(
        self, variable_idx: int, update_domain: Callable, processor_idx: int, solution_queue: Queue
    ) -> None:
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs = get_function_addresses()
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
                self.stacks_height,
                self.triggered_propagators,
                self.consistency_algorithm_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            self.statistics[STATS_IDX_OPTIMIZER_SOLUTION_NB] += 1
            solution_queue.put((processor_idx, solution, self.statistics))
            reset(
                self.problem,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.stacks_height,
                self.triggered_propagators,
            )
            update_domain(
                self.shr_domains_stack[0],
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                variable_idx,
                solution[variable_idx],
            )
        solution_queue.put((processor_idx, None, self.statistics))

    def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
        compute_domains_addrs, var_heuristic_addrs, dom_heuristic_addrs = get_function_addresses()
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
                self.stacks_height,
                self.triggered_propagators,
                self.consistency_algorithm_idx,
                self.var_heuristic_idx,
                self.dom_heuristic_idx,
                compute_domains_addrs,
                var_heuristic_addrs,
                dom_heuristic_addrs,
            )
            if solution is None:
                break
            solution_queue.put((processor_idx, solution, self.statistics))
            if not backtrack(
                self.statistics,
                self.shr_domains_stack,
                self.not_entailed_propagators_stack,
                self.stacks_height,
                self.triggered_propagators,
            ):
                break
        solution_queue.put((processor_idx, None, self.statistics))


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    stacks_height: NDArray,
    triggered_propagators: NDArray,
) -> bool:
    """
    Backtracks and updates the problem's domains
    :return: true iff it is possible to backtrack
    """
    if stacks_height[0] == 1:
        return False
    stacks_height[0] -= 1
    shr_domains_stack[0, :, :] = shr_domains_stack[stacks_height[0], :, :]
    not_entailed_propagators_stack[0, :] = not_entailed_propagators_stack[stacks_height[0], :]
    statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
    triggered_propagators[:] = not_entailed_propagators_stack[0, :]  # numba does not support copyto
    return True


def reset(
    problem: Problem,
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    stacks_height: NDArray,
    triggered_propagators: NDArray,
) -> None:
    shr_domains_stack[0] = problem.shr_domains_lst
    not_entailed_propagators_stack[0, :] = True
    stacks_height[0] = 1
    triggered_propagators.fill(True)


@njit(cache=True)
def make_choice(
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
    stacks_height: NDArray,
    triggered_propagators: NDArray,
    consistency_algorithm_idx: int,
    var_heuristic_idx: int,
    dom_heuristic_idx: int,
    compute_domains_addrs: NDArray,
    var_heuristic_addrs: NDArray,
    dom_heuristic_addrs: NDArray,
) -> int:
    """
    Makes a choice and returns a status
    :return: the status as an integer
    """
    # first filter
    while True:
        status = consistency_algorithm(
            consistency_algorithm_idx,
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
            shr_domains_stack[0],
            not_entailed_propagators_stack[0],
            triggered_propagators,
            compute_domains_addrs,
        )
        if status != PROBLEM_INCONSISTENT:
            break
        if not backtrack(
            statistics, shr_domains_stack, not_entailed_propagators_stack, stacks_height, triggered_propagators
        ):
            return PROBLEM_INCONSISTENT
    if status == PROBLEM_SOLVED:
        statistics[STATS_IDX_SOLVER_SOLUTION_NB] += 1
        return PROBLEM_SOLVED
    # then make a choice
    var_heuristic_function = (
        VAR_HEURISTIC_FCTS[var_heuristic_idx]
        if NUMBA_DISABLE_JIT
        else function_from_address(TYPE_VAR_HEURISTIC, var_heuristic_addrs[var_heuristic_idx])
    )
    dom_idx = var_heuristic_function(shr_domains_stack[0])
    shr_domains_stack[stacks_height[0], :, :] = shr_domains_stack[0, :, :]
    not_entailed_propagators_stack[stacks_height[0], :] = not_entailed_propagators_stack[0, :]
    stacks_height[0] += 1
    dom_heuristic_function = (
        DOM_HEURISTIC_FCTS[dom_heuristic_idx]
        if NUMBA_DISABLE_JIT
        else function_from_address(TYPE_DOM_HEURISTIC, dom_heuristic_addrs[dom_heuristic_idx])
    )
    event = dom_heuristic_function(shr_domains_stack[0, dom_idx], shr_domains_stack[stacks_height[0] - 1, dom_idx])
    np.logical_or(triggered_propagators, shr_domains_propagators[dom_idx, event], triggered_propagators)
    statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
    if stacks_height[0] - 1 > statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]:
        statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = stacks_height[0] - 1
    return PROBLEM_TO_FILTER


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
    stacks_height: NDArray,
    triggered_propagators: NDArray,
    consistency_algorithm_idx: int,
    var_heuristic_idx: int,
    dom_heuristic_idx: int,
    compute_domains_addrs: NDArray,
    var_heuristic_addrs: NDArray,
    dom_heuristic_addrs: NDArray,
) -> Optional[NDArray]:
    """
    Find at most one solution.
    :return: the solution if it exists or None
    """
    while True:
        status = make_choice(
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
            stacks_height,
            triggered_propagators,
            consistency_algorithm_idx,
            var_heuristic_idx,
            dom_heuristic_idx,
            compute_domains_addrs,
            var_heuristic_addrs,
            dom_heuristic_addrs,
        )
        if status != PROBLEM_TO_FILTER:
            break
    return get_solution(shr_domains_stack[0], dom_indices_arr, dom_offsets_arr) if status == PROBLEM_SOLVED else None
