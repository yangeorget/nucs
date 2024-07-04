from typing import Iterator, List, Optional

from numpy.typing import NDArray

from ncs.heuristics.first_variable_heuristic import FirstVariableHeuristic
from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problems.problem import Problem, is_not_solved
from ncs.solvers.solver import Solver
from ncs.utils import (
    STATS_SOLVER_BACKTRACKS_NB,
    STATS_SOLVER_CHOICES_NB,
    STATS_SOLVER_CP_MAX,
    STATS_SOLVER_SOLUTIONS_NB,
)


class BacktrackSolver(Solver):
    def __init__(self, problem: Problem, heuristic: Heuristic = FirstVariableHeuristic(MinValueHeuristic())):
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
        if not self.filter():
            return None
        while is_not_solved(self.problem.shared_domains):
            changes = self.heuristic.choose(self.choice_points, self.problem)
            self.statistics[STATS_SOLVER_CHOICES_NB] += 1
            value = len(self.choice_points)
            self.statistics[STATS_SOLVER_CP_MAX] = max(self.statistics[STATS_SOLVER_CP_MAX], value)
            if not self.filter(changes):
                return None
        return self.problem.get_values()  # problem is solved

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
