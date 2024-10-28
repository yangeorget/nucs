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
from numpy.typing import NDArray

from nucs.constants import PROBLEM_INCONSISTENT, PROBLEM_SOLVED, PROBLEM_TO_FILTER
from nucs.examples.golomb.golomb_problem import golomb_consistency_algorithm
from nucs.numpy import new_not_entailed_propagators, new_shr_domains_by_values, new_triggered_propagators
from nucs.problems.problem import Problem
from nucs.propagators.propagators import COMPUTE_DOMAINS_ADDRS
from nucs.solvers.choice_points import ChoicePoints
from nucs.solvers.consistency_algorithms import bound_consistency_algorithm
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
        var_heuristic: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        dom_heuristic: int = DOM_HEURISTIC_MIN_VALUE,
        consistency_algorithm: int = 0,
    ):
        """
        Inits the solver.
        :param problem: the problem
        :param var_heuristic: a heuristic for selecting a variable/domain
        :param dom_heuristic: a heuristic for reducing a domain
        """
        self.var_heuristic = var_heuristic
        self.dom_heuristic = dom_heuristic
        self.consistency_algorithm = consistency_algorithm
        self.triggered_propagators = new_triggered_propagators(problem.propagator_nb)
        problem.init()
        self.problem = problem
        shr_domains_arr = new_shr_domains_by_values(self.problem.shr_domains_lst)
        not_entailed_propagators = new_not_entailed_propagators(self.problem.propagator_nb)
        self.choice_points = ChoicePoints()
        self.choice_points.put((shr_domains_arr, not_entailed_propagators))
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
        best_solution = None
        while (
            solution := solve_one(
                self.statistics,
                self.problem,
                self.choice_points,
                self.triggered_propagators,
                self.var_heuristic,
                self.dom_heuristic,
                self.consistency_algorithm,
            )
        ) is not None:
            best_solution = solution
            self.statistics[STATS_IDX_OPTIMIZER_SOLUTION_NB] += 1
            reset(self.problem, self.choice_points, self.triggered_propagators)
            update_domain(
                self.choice_points.get_shr_domains(),
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
        while (
            solution := solve_one(
                self.statistics,
                self.problem,
                self.choice_points,
                self.triggered_propagators,
                self.consistency_algorithm,
                self.var_heuristic,
                self.dom_heuristic,
            )
        ) is not None:
            yield solution
            if not backtrack(self.statistics, self.choice_points, self.triggered_propagators):
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
        while (
            solution := solve_one(
                self.statistics,
                self.problem,
                self.choice_points,
                self.triggered_propagators,
                self.consistency_algorithm,
                self.var_heuristic,
                self.dom_heuristic,
            )
        ) is not None:
            self.statistics[STATS_IDX_OPTIMIZER_SOLUTION_NB] += 1
            solution_queue.put((processor_idx, solution, self.statistics))
            reset(self.problem, self.choice_points, self.triggered_propagators)
            update_domain(
                self.choice_points.get_shr_domains(),
                self.problem.dom_indices_arr,
                self.problem.dom_offsets_arr,
                variable_idx,
                solution[variable_idx],
            )
        solution_queue.put((processor_idx, None, self.statistics))

    def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
        while (
            solution := solve_one(
                self.statistics,
                self.problem,
                self.choice_points,
                self.triggered_propagators,
                self.consistency_algorithm,
                self.var_heuristic,
                self.dom_heuristic,
            )
        ) is not None:
            solution_queue.put((processor_idx, solution, self.statistics))
            if not backtrack(self.statistics, self.choice_points, self.triggered_propagators):
                break
        solution_queue.put((processor_idx, None, self.statistics))


def backtrack(statistics: NDArray, choice_points: ChoicePoints, triggered_propagators: NDArray) -> bool:
    """
    Backtracks and updates the problem's domains
    :return: true iff it is possible to backtrack
    """
    if not choice_points.pop():
        return False
    statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
    np.copyto(triggered_propagators, choice_points.get_not_entailed_propagators())
    return True


def reset(problem: Problem, choice_points: ChoicePoints, triggered_propagators: NDArray) -> None:
    choice_points.clear()
    choice_points.put(
        (new_shr_domains_by_values(problem.shr_domains_lst), new_not_entailed_propagators(problem.propagator_nb))
    )
    triggered_propagators.fill(True)


def make_choice(
    statistics: NDArray,
    problem: Problem,
    choice_points: ChoicePoints,
    triggered_propagators: NDArray,
    var_heuristic: int,
    dom_heuristic: int,
    consistency_algorithm: int,
) -> int:
    """
    Makes a choice and returns a status
    :return: the status as an integer
    """
    # first filter
    while (
        status := (
            bound_consistency_algorithm(
                statistics,
                problem.algorithms,
                problem.var_bounds,
                problem.param_bounds,
                problem.dom_indices_arr,
                problem.dom_offsets_arr,
                problem.props_dom_indices,
                problem.props_dom_offsets,
                problem.props_parameters,
                problem.shr_domains_propagators,
                choice_points.get_shr_domains(),
                choice_points.get_not_entailed_propagators(),
                triggered_propagators,
                COMPUTE_DOMAINS_ADDRS,
            )
            if consistency_algorithm == 0
            else golomb_consistency_algorithm(
                statistics,
                problem.algorithms,
                problem.var_bounds,
                problem.param_bounds,
                problem.dom_indices_arr,
                problem.dom_offsets_arr,
                problem.props_dom_indices,
                problem.props_dom_offsets,
                problem.props_parameters,
                problem.shr_domains_propagators,
                choice_points.get_shr_domains(),
                choice_points.get_not_entailed_propagators(),
                triggered_propagators,
                COMPUTE_DOMAINS_ADDRS,
            )
        )
    ) == PROBLEM_INCONSISTENT:
        if not backtrack(statistics, choice_points, triggered_propagators):
            return PROBLEM_INCONSISTENT
    if status == PROBLEM_SOLVED:
        statistics[STATS_IDX_SOLVER_SOLUTION_NB] += 1
        return PROBLEM_SOLVED
    # then make a choice
    shr_domains_arr = choice_points.get_shr_domains()
    not_entailed_propagators = choice_points.get_not_entailed_propagators()
    var_heuristic_function = (
        VAR_HEURISTIC_FCTS[var_heuristic]
        # if NUMBA_DISABLE_JIT
        # else function_from_address(VAR_HEURISTIC_TYPE, VAR_HEURISTIC_ADDRS[var_heuristic])
    )
    dom_idx = var_heuristic_function(shr_domains_arr)
    shr_domains_copy = shr_domains_arr.copy(order="F")
    not_entailed_propagators_copy = not_entailed_propagators.copy()
    choice_points.put((shr_domains_copy, not_entailed_propagators_copy))
    dom_heuristic_function = (
        DOM_HEURISTIC_FCTS[dom_heuristic]
        # if NUMBA_DISABLE_JIT
        # else function_from_address(DOM_HEURISTIC_TYPE, DOM_HEURISTIC_ADDRS[dom_heuristic])
    )
    event = dom_heuristic_function(shr_domains_arr[dom_idx], shr_domains_copy[dom_idx])
    np.logical_or(
        triggered_propagators,
        problem.shr_domains_propagators[dom_idx, event],
        triggered_propagators,
    )
    statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
    cp_max_depth = choice_points.size() - 1
    if cp_max_depth > statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]:
        statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = cp_max_depth
    return PROBLEM_TO_FILTER


def solve_one(
    statistics: NDArray,
    problem: Problem,
    choice_points: ChoicePoints,
    triggered_propagators: NDArray,
    var_heuristic: int,
    dom_heuristic: int,
    consistency_algorithm: int,
) -> Optional[NDArray]:
    """
    Find at most one solution.
    :return: the solution if it exists or None
    """
    while (
        status := make_choice(
            statistics,
            problem,
            choice_points,
            triggered_propagators,
            var_heuristic,
            dom_heuristic,
            consistency_algorithm,
        )
    ) == PROBLEM_TO_FILTER:
        pass
    return (
        get_solution(choice_points.get_shr_domains(), problem.dom_indices_arr, problem.dom_offsets_arr)
        if status == PROBLEM_SOLVED
        else None
    )
