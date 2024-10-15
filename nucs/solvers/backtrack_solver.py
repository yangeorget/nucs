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
from multiprocessing import Manager
from multiprocessing.managers import ListProxy
from threading import Event
from typing import Any, Callable, Iterator, List, Optional

import numpy as np

from nucs.constants import MIN, PROBLEM_INCONSISTENT, PROBLEM_SOLVED
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
    init_statistics, get_statistics,
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

    def solve(self) -> Iterator[List[int]]:
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        manager = Manager()
        run = manager.Event()
        run.set()
        solution_proxy = manager.list()
        while True:
            self.solve_one(run, solution_proxy)
            if len(solution_proxy) == 0:
                break
            yield list(solution_proxy)
            run.set()
            solution_proxy[:] = []
            if not self.backtrack():
                break

    def solve_one(self, run: Event, solution_proxy: ListProxy) -> None:
        if not self.problem.ready:
            self.problem.init_problem(self.statistics)
            self.problem.ready = True
        while run.is_set():
            while (status := self.consistency_algorithm(self.statistics, self.problem)) == PROBLEM_INCONSISTENT:
                if not self.backtrack():
                    run.clear()
                    return
            if status == PROBLEM_SOLVED:
                self.statistics[STATS_SOLVER_SOLUTION_NB] += 1
                values = self.problem.shr_domains_arr[self.problem.dom_indices_arr, MIN] + self.problem.dom_offsets_arr
                solution_proxy.extend(values.tolist())
                run.clear()
                return
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

    def minimize(self, variable_idx: int) -> Optional[List[int]]:
        manager = Manager()
        run = manager.Event()
        run.set()
        solution_proxy = manager.list()
        best_solution = []
        while True:
            self.solve_one(run, solution_proxy)
            if len(solution_proxy) == 0:
                break
            best_solution = list(solution_proxy)
            run.set()
            solution_proxy[:] = []
            self.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
            self.reset()
            self.problem.set_max_value(variable_idx, best_solution[variable_idx] - 1)
        return best_solution

    def maximize(self, variable_idx: int) -> Optional[List[int]]:
        manager = Manager()
        run = manager.Event()
        run.set()
        solution_proxy = manager.list()
        best_solution = []
        while True:
            self.solve_one(run, solution_proxy)
            if len(solution_proxy) == 0:
                break
            best_solution = list(solution_proxy)
            run.set()
            solution_proxy[:] = []
            self.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
            self.reset()
            self.problem.set_min_value(variable_idx, best_solution[variable_idx] + 1)
        return best_solution

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
