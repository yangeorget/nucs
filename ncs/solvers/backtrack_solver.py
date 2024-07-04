from typing import Iterator, List, Optional

from numpy.typing import NDArray

from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.variable_heuristic import VariableHeuristic, first_not_instantiated_variable_heuristic, \
    min_value_domain_heuristic
from ncs.problems.problem import Problem, is_solved
from ncs.solvers.solver import Solver
from ncs.utils import (
    STATS_SOLVER_BACKTRACKS_NB,
    STATS_SOLVER_CHOICES_NB,
    STATS_SOLVER_CP_MAX,
    STATS_SOLVER_SOLUTIONS_NB,
)


class BacktrackSolver(Solver):
    def __init__(self, problem: Problem, heuristic: Heuristic = VariableHeuristic(first_not_instantiated_variable_heuristic, min_value_domain_heuristic)):
        super().__init__(problem)
        self.choice_points = []  # type: ignore
        self.heuristic = heuristic

    def solve(self) -> Iterator[List[int]]:
        while True:
            solution = self.solve_one()
            if solution is None:
                break
            self.statistics[STATS_SOLVER_SOLUTIONS_NB] += 1
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[List[int]]:
        changes = None
        while self.filter(changes):
            if is_solved(self.problem.shared_domains):
                return self.problem.get_values()  # problem is solved
            changes = self.heuristic.choose(self.choice_points, self.problem)
            self.statistics[STATS_SOLVER_CHOICES_NB] += 1
            self.statistics[STATS_SOLVER_CP_MAX] = max(self.statistics[STATS_SOLVER_CP_MAX], len(self.choice_points))
        return None

    def backtrack(self) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        if len(self.choice_points) == 0:
            return False
        self.statistics[STATS_SOLVER_BACKTRACKS_NB] += 1
        self.problem.shared_domains = self.choice_points.pop()
        return True

    def filter(self, changes: Optional[NDArray] = None) -> bool:
        while not self.problem.filter(self.statistics, changes):
            if not self.backtrack():
                return False
        return True
