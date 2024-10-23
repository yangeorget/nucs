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
from nucs.problems.problem import Problem
from nucs.solvers.choice_points import ChoicePointList, ChoicePoints
from nucs.solvers.consistency_algorithms import bound_consistency_algorithm
from nucs.solvers.heuristics import first_not_instantiated_var_heuristic, min_value_dom_heuristic
from nucs.solvers.solver import Solver
from nucs.statistics import (
    STATS_IDX_OPTIMIZER_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_SOLUTION_NB,
    init_statistics,
)


def decrease_max(problem: Problem, var_idx: int, value: int) -> None:
    """
    Decreases the max of a variable
    :param problem: the problem
    :param var_idx: the index of the variable
    :param value: the current max
    """
    problem.set_max_value(var_idx, value - 1)


def increase_min(problem: Problem, var_idx: int, value: int) -> None:
    """
    Increases the min of a variable
    :param problem: the problem
    :param var_idx: the index of the variable
    :param value: the current min
    """
    problem.set_min_value(var_idx, value + 1)


class BacktrackSolver(Solver):
    """
    A solver relying on a backtracking mechanism.
    """

    def __init__(
        self,
        problem: Problem,
        consistency_algorithm: Callable = bound_consistency_algorithm,
        var_heuristic: Callable = first_not_instantiated_var_heuristic,
        dom_heuristic: Callable = min_value_dom_heuristic,
    ):
        """
        Inits the solver.
        :param problem: the problem
        :param consistency_algorithm: a consistency algorithm (usually bound consistency)
        :param var_heuristic: a heuristic for selecting a variable/domain
        :param dom_heuristic: a heuristic for reducing a domain
        """
        self.problem = problem
        self.statistics = init_statistics()
        self.consistency_algorithm = consistency_algorithm
        self.var_heuristic = var_heuristic
        self.dom_heuristic = dom_heuristic

    def init_problem(self) -> None:
        """
        Inits the problem.
        """
        self.problem.init(self.statistics)

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

    def optimize(self, variable_idx: int, update_target_domain: Callable) -> Optional[NDArray]:
        self.init_problem()
        choice_points = ChoicePointList()
        best_solution = None
        while (solution := self.solve_one(choice_points)) is not None:
            best_solution = solution
            self.statistics[STATS_IDX_OPTIMIZER_SOLUTION_NB] += 1
            self.reset(choice_points)
            update_target_domain(self.problem, variable_idx, best_solution[variable_idx])
        return best_solution

    def solve(self) -> Iterator[NDArray]:
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        self.init_problem()
        choice_points = ChoicePointList()
        while (solution := self.solve_one(choice_points)) is not None:
            yield solution
            if not self.backtrack(choice_points):
                break

    def solve_one(self, choice_points: ChoicePoints) -> Optional[NDArray]:
        """
        Find at most one solution.
        :return: the solution if it exists or None
        """
        while (status := self.make_choice(choice_points)) == PROBLEM_TO_FILTER:
            pass
        return self.problem.get_solution() if status == PROBLEM_SOLVED else None

    def make_choice(self, choice_points: ChoicePoints) -> int:
        """
        Makes a choice and returns a status
        :return: the status as an integer
        """
        # first filter
        while (status := self.consistency_algorithm(self.statistics, self.problem)) == PROBLEM_INCONSISTENT:
            if not self.backtrack(choice_points):
                return PROBLEM_INCONSISTENT
        if status == PROBLEM_SOLVED:
            self.statistics[STATS_IDX_SOLVER_SOLUTION_NB] += 1
            return PROBLEM_SOLVED
        # then make a choice
        dom_idx = self.var_heuristic(self.problem.shr_domains_arr)
        shr_domains_copy = self.problem.shr_domains_arr.copy(order="F")
        not_entailed_propagators_copy = self.problem.not_entailed_propagators.copy()
        choice_points.put((shr_domains_copy, not_entailed_propagators_copy))
        event = self.dom_heuristic(self.problem.shr_domains_arr[dom_idx], shr_domains_copy[dom_idx])
        np.logical_or(
            self.problem.triggered_propagators,
            self.problem.shr_domains_propagators[dom_idx, event],
            self.problem.triggered_propagators,
        )
        self.statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
        cp_max_depth = choice_points.size()
        if cp_max_depth > self.statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]:
            self.statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = cp_max_depth
        return PROBLEM_TO_FILTER

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
        self, variable_idx: int, update_target_domain: Callable, processor_idx: int, solution_queue: Queue
    ) -> None:
        self.init_problem()
        choice_points = ChoicePointList()
        while (solution := self.solve_one(choice_points)) is not None:
            self.statistics[STATS_IDX_OPTIMIZER_SOLUTION_NB] += 1
            solution_queue.put((processor_idx, solution, self.statistics))
            self.reset(choice_points)
            update_target_domain(self.problem, variable_idx, solution[variable_idx])
        solution_queue.put((processor_idx, None, self.statistics))

    def solve_and_queue(self, processor_idx: int, solution_queue: Queue) -> None:
        self.init_problem()
        choice_points = ChoicePointList()
        while (solution := self.solve_one(choice_points)) is not None:
            solution_queue.put((processor_idx, solution, self.statistics))
            if not self.backtrack(choice_points):
                break
        solution_queue.put((processor_idx, None, self.statistics))

    def backtrack(self, choice_points: ChoicePoints) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        if choice_points.is_empty():
            return False
        self.statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
        self.problem.reset(choice_points.get())  # TODO: optimize by reusing
        return True

    def reset(self, choice_points: ChoicePoints) -> None:
        """
        Resets the solver by resetting the problem and the choice points.
        """
        choice_points.clear()
        self.problem.reset()
