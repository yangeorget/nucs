from typing import Callable, Iterator, List, Optional

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
        super().__init__(problem)
        self.choice_points = []  # type: ignore
        self.consistency_algorithm = consistency_algorithm
        self.var_heuristic = var_heuristic
        self.dom_heuristic = dom_heuristic

    def solve(self) -> Iterator[List[int]]:
        while (solution := self.solve_one()) is not None:
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[List[int]]:
        """
        Find at most one solution.
        :return: the solution if it exists or None
        """
        if not self.problem.ready:
            self.problem.init_problem(self.statistics)
            self.problem.ready = True
        while True:
            while (status := self.consistency_algorithm(self.statistics, self.problem)) == PROBLEM_INCONSISTENT:
                if not self.backtrack():
                    return None
            if status == PROBLEM_SOLVED:
                self.statistics[STATS_SOLVER_SOLUTION_NB] += 1
                values = self.problem.shr_domains_arr[self.problem.dom_indices_arr, MIN] + self.problem.dom_offsets_arr
                return values.tolist()
            dom_idx = self.var_heuristic(self.problem.shr_domains_arr)
            shr_domains_copy = self.problem.shr_domains_arr.copy(order="F")
            self.choice_points.append(shr_domains_copy)
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
        solution = None
        while (new_solution := self.solve_one()) is not None:
            solution = new_solution
            self.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
            self.reset()
            self.problem.set_max_value(variable_idx, solution[variable_idx] - 1)
        return solution

    def maximize(self, variable_idx: int) -> Optional[List[int]]:
        solution = None
        while (new_solution := self.solve_one()) is not None:
            solution = new_solution
            self.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
            self.reset()
            self.problem.set_min_value(variable_idx, solution[variable_idx] + 1)
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
