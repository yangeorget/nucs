from typing import Iterator, List, Optional

from nucs.constants import MIN, PROBLEM_INCONSISTENT, PROBLEM_SOLVED
from nucs.numpy import new_shr_domain_changes
from nucs.problems.problem import Problem
from nucs.solvers.heuristics import (
    DOM_HEURISTIC_FCTS,
    DOM_HEURISTIC_MIN_VALUE,
    VAR_HEURISTIC_FCTS,
    VAR_HEURISTIC_FIRST_NON_INSTANTIATED,
)
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
        variable_heuristic: int = VAR_HEURISTIC_FIRST_NON_INSTANTIATED,
        domain_heuristic: int = DOM_HEURISTIC_MIN_VALUE,
    ):
        super().__init__(problem)
        self.choice_points = []  # type: ignore
        self.shr_domain_changes = new_shr_domain_changes(len(self.problem.shr_domains_lst))
        self.variable_heuristic = variable_heuristic
        self.domain_heuristic = domain_heuristic

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
        while True:
            while (status := self.problem.filter(self.shr_domain_changes)) == PROBLEM_INCONSISTENT:
                if not self.backtrack():
                    return None
            if status == PROBLEM_SOLVED:
                self.problem.statistics[STATS_SOLVER_SOLUTION_NB] += 1
                values = self.problem.shr_domains_arr[self.problem.dom_indices_arr, MIN] + self.problem.dom_offsets_arr
                return values.tolist()
            shr_domains_copy = self.problem.shr_domains_arr.copy(order="F")
            var_idx = VAR_HEURISTIC_FCTS[self.variable_heuristic](
                self.problem.shr_domains_arr, self.problem.dom_indices_arr
            )
            shr_domain_idx = self.problem.dom_indices_arr[var_idx]
            DOM_HEURISTIC_FCTS[self.domain_heuristic](
                self.problem.shr_domains_arr,
                self.shr_domain_changes,
                shr_domains_copy,
                shr_domain_idx,
            )
            self.choice_points.append(shr_domains_copy)
            self.problem.statistics[STATS_SOLVER_CHOICE_NB] += 1
            self.problem.statistics[STATS_SOLVER_CHOICE_DEPTH] = max(
                self.problem.statistics[STATS_SOLVER_CHOICE_DEPTH], len(self.choice_points)
            )

    def minimize(self, variable_idx: int) -> Optional[List[int]]:
        solution = None
        while (new_solution := self.solve_one()) is not None:
            solution = new_solution
            self.problem.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
            self.reset()
            self.problem.set_max_value(variable_idx, solution[variable_idx] - 1)
        return solution

    def maximize(self, variable_idx: int) -> Optional[List[int]]:
        solution = None
        while (new_solution := self.solve_one()) is not None:
            solution = new_solution
            self.problem.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
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
        self.problem.statistics[STATS_SOLVER_BACKTRACK_NB] += 1
        self.shr_domain_changes.fill(True)
        self.problem.shr_domains_arr = self.choice_points.pop()
        self.problem.reset_not_entailed_propagators()
        return True

    def reset(self) -> None:
        """
        Resets the solver by resetting the problem and the choice points.
        """
        self.choice_points.clear()
        self.shr_domain_changes.fill(True)
        self.problem.reset_shr_domains()
        self.problem.reset_not_entailed_propagators()
