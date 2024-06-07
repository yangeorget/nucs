from typing import Iterator, Optional

from numpy.typing import NDArray

from ncs.heuristics.first_variable_heuristic import FirstVariableHeuristic
from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problems.problem import Problem
from ncs.solvers.solver import Solver


class BacktrackSolver(Solver):
    def __init__(self, problem: Problem, heuristic: Heuristic = FirstVariableHeuristic(MinValueHeuristic())):
        super().__init__(problem)
        self.choice_points = []  # type: ignore
        self.heuristic = heuristic
        self.statistics["backtracksolver.backtracks.nb"] = 0
        self.statistics["backtracksolver.choicepoints.max"] = 0

    def solve(self) -> Iterator[NDArray]:
        while True:
            solution = self.solve_one()
            if solution is None:
                break
            self.statistics["solver.solutions.nb"] += 1
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[NDArray]:
        if not self.filter():
            return None
        while self.problem.is_not_solved():
            changes = None
            self.heuristic.choose(self.choice_points, self.problem)  # TODO: make choice should update changes
            self.statistics["backtracksolver.choicepoints.max"] = max(
                len(self.choice_points), self.statistics["backtracksolver.choicepoints.max"]
            )
            if not self.filter(changes):
                return None
        # problem is solved
        return self.problem.get_domains()

    def backtrack(self) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        if len(self.choice_points) == 0:
            return False
        self.statistics["backtracksolver.backtracks.nb"] += 1
        self.problem.shr_domains = self.choice_points.pop()
        return True

    def filter(self, changes: Optional[NDArray] = None) -> bool:
        while not self.problem.filter(changes, self.statistics):
            # filtering has detected an inconsistency
            if not self.backtrack():
                # backtracking is not feasible
                return False
        return True
