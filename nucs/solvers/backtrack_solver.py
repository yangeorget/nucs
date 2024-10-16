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
from typing import Callable, Iterator, List, Optional

import numpy as np

from nucs.constants import PROBLEM_INCONSISTENT, PROBLEM_SOLVED, PROBLEM_TO_FILTER
from nucs.problems.problem import Problem
from nucs.solvers.consistency_algorithms import bound_consistency_algorithm
from nucs.solvers.heuristics import first_not_instantiated_var_heuristic, min_value_dom_heuristic
from nucs.solvers.solver import Solver
from nucs.statistics import (
    STATS_OPTIMIZER_SOLUTION_NB,
    STATS_SOLVER_BACKTRACK_NB,
    STATS_SOLVER_CHOICE_DEPTH,
    STATS_SOLVER_CHOICE_NB,
    STATS_SOLVER_SOLUTION_NB,
    init_statistics,
)


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
        self.choice_points = []  # type: ignore
        self.consistency_algorithm = consistency_algorithm
        self.var_heuristic = var_heuristic
        self.dom_heuristic = dom_heuristic

    def init_problem(self):
        self.problem.init(self.statistics)

    def solve(self) -> Iterator[List[int]]:
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        self.init_problem()
        while (solution := self.solve_one()) is not None:
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[List[int]]:
        """
        Find at most one solution.
        :return: the solution if it exists or None
        """
        while (status := self.make_choice()) == PROBLEM_TO_FILTER:
            pass
        return self.problem.get_solution() if status == PROBLEM_SOLVED else None

    def solve_all_queue(self, idx: int, queue: Queue) -> None:
        # TODO: use a a semaphor or event
        self.solve_all(lambda solution: queue.put((idx, solution, self.statistics)))


    def make_choice(self) -> int:
        # first filter
        while (status := self.consistency_algorithm(self.statistics, self.problem)) == PROBLEM_INCONSISTENT:
            if not self.backtrack():
                return PROBLEM_INCONSISTENT
        if status == PROBLEM_SOLVED:
            self.statistics[STATS_SOLVER_SOLUTION_NB] += 1
            return PROBLEM_SOLVED
        # then make a choice
        dom_idx = self.var_heuristic(self.problem.shr_domains_arr)
        shr_domains_copy = self.problem.shr_domains_arr.copy(order="F")
        not_entailed_propagators_copy = self.problem.not_entailed_propagators.copy()
        self.choice_points.append((shr_domains_copy, not_entailed_propagators_copy))
        event = self.dom_heuristic(self.problem.shr_domains_arr[dom_idx], shr_domains_copy[dom_idx])
        np.logical_or(
            self.problem.triggered_propagators,
            self.problem.shr_domains_propagators[dom_idx, event],
            self.problem.triggered_propagators,
        )
        self.statistics[STATS_SOLVER_CHOICE_NB] += 1
        cp_max_depth = len(self.choice_points)
        if cp_max_depth > self.statistics[STATS_SOLVER_CHOICE_DEPTH]:
            self.statistics[STATS_SOLVER_CHOICE_DEPTH] = cp_max_depth
        return PROBLEM_TO_FILTER

    def minimize(self, variable_idx: int) -> Optional[List[int]]:
        return self.optimize(variable_idx, lambda var_idx, value: self.problem.set_max_value(var_idx, value - 1))

    def maximize(self, variable_idx: int) -> Optional[List[int]]:
        return self.optimize(variable_idx, lambda var_idx, value: self.problem.set_min_value(var_idx, value + 1))

    def optimize(self, variable_idx: int, update_target_domain: Callable) -> Optional[List[int]]:
        self.init_problem()
        solution = None
        while (new_solution := self.solve_one()) is not None:
            solution = new_solution
            self.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
            self.reset()
            update_target_domain(variable_idx, solution[variable_idx])
        return solution

    def backtrack(self) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        if len(self.choice_points) == 0:
            return False
        self.statistics[STATS_SOLVER_BACKTRACK_NB] += 1
        self.problem.reset(self.choice_points.pop())  # TODO: optimize by reusing
        return True

    def reset(self) -> None:
        """
        Resets the solver by resetting the problem and the choice points.
        """
        self.choice_points.clear()
        self.problem.reset()
