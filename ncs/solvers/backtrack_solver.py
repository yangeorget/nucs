from typing import Iterator, List, Optional

from numpy.typing import NDArray

from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.variable_heuristic import (
    VariableHeuristic,
    first_not_instantiated_variable_heuristic,
    min_value_domain_heuristic,
)
from ncs.problems.problem import Problem, is_solved
from ncs.solvers.solver import Solver
from ncs.utils import (
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
        heuristic: Heuristic = VariableHeuristic(first_not_instantiated_variable_heuristic, min_value_domain_heuristic),
    ):
        super().__init__(problem)
        self.choice_points = []  # type: ignore
        self.heuristic = heuristic

    def solve(self) -> Iterator[List[int]]:
        while True:
            solution = self.solve_one()
            if solution is None:
                break
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[List[int]]:
        changes = None
        while self.filter(changes):
            if is_solved(self.problem.shared_domains):
                self.statistics[STATS_SOLVER_SOLUTION_NB] += 1
                return self.problem.get_values()  # problem is solved
            domains, changes = self.heuristic.choose(self.problem.shared_domains, self.problem.domain_indices)
            self.choice_points.append(domains)
            self.statistics[STATS_SOLVER_CHOICE_NB] += 1
            self.statistics[STATS_SOLVER_CHOICE_DEPTH] = max(
                self.statistics[STATS_SOLVER_CHOICE_DEPTH], len(self.choice_points)
            )
        return None

    def minimize(self, variable_idx: int) -> Optional[List[int]]:
        solution = None
        while (new_solution := self.solve_one()) is not None:
            solution = new_solution
            self.statistics[STATS_OPTIMIZER_SOLUTION_NB] += 1
            self.reset()
            self.problem.set_max_value(variable_idx, solution[variable_idx] - 1)
        return solution

    def backtrack(self) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        if len(self.choice_points) == 0:
            return False
        self.statistics[STATS_SOLVER_BACKTRACK_NB] += 1
        self.problem.shared_domains = self.choice_points.pop()
        return True

    def filter(self, changes: Optional[NDArray] = None) -> bool:
        while not self.problem.filter(self.statistics, changes):
            if not self.backtrack():
                return False
        return True

    def reset(self) -> None:
        self.choice_points = []
        self.problem.reset()
