from typing import Iterator, List, Optional

from numpy.typing import NDArray

from ncs.heuristics.first_variable_heuristic import FirstVariableHeuristic
from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problems.problem import Problem
from ncs.solvers.solver import Solver
from ncs.utils import (
    STATS_SOLVER_BACKTRACKS_NB,
    STATS_SOLVER_CHOICES_NB,
    STATS_SOLVER_CP_MAX,
    STATS_SOLVER_SOLUTIONS_NB,
    stats_inc,
    stats_max,
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
            stats_inc(self.statistics, STATS_SOLVER_SOLUTIONS_NB)
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[List[int]]:
        if not self.filter():
            return None
        while self.problem.is_not_solved():
            changes = self.heuristic.choose(self.choice_points, self.problem)
            stats_inc(self.statistics, STATS_SOLVER_CHOICES_NB)
            stats_max(self.statistics, STATS_SOLVER_CP_MAX, len(self.choice_points))
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
        stats_inc(self.statistics, STATS_SOLVER_BACKTRACKS_NB)
        self.problem.shr_domains = self.choice_points.pop()
        return True

    def filter(self, changes: Optional[NDArray] = None) -> bool:
        while not self.problem.filter(self.statistics, changes):
            if not self.backtrack():
                return False
        return True
