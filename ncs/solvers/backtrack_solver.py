from typing import Iterator, Optional

from numpy.typing import NDArray

from ncs.heuristics.first_variable_heuristic import FirstVariableHeuristic
from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.min_value_heuristic import MinValueHeuristic
from ncs.problem import Problem
from ncs.solvers.solver import Solver


class BacktrackSolver(Solver):
    def __init__(self, problem: Problem, heuristic: Heuristic = FirstVariableHeuristic(MinValueHeuristic())):
        super().__init__(problem)
        self.choice_points = []  # type: ignore
        self.heuristic = heuristic
        self.statistics["backtracksolver.backtracks.nb"] = 0
        self.statistics["backtracksolver.choicepoints.max"] = 0

    def solve(self) -> Iterator[NDArray]:
        # print("solve()")
        while True:
            solution = self.solve_one()
            if solution is None:
                break
            self.statistics["solver.solutions.nb"] += 1
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[NDArray]:
        if not self.problem.filter(self.statistics):  # TODO: filter on everything
            return None
        while not self.problem.is_solved():
            if self.heuristic.make_choice(self.choice_points, self.problem):  # let's make a choice
                if len(self.choice_points) >= self.statistics["backtracksolver.choicepoints.max"]:
                    self.statistics["backtracksolver.choicepoints.max"] = len(self.choice_points)
            while not self.problem.filter(self.statistics):  # the choice was not consistent
                # TODO: filter on everything or on choice
                if not self.backtrack():
                    return None
        return self.problem.domains

    def backtrack(self) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        # print("backtrack()")
        if len(self.choice_points) == 0:
            return False
        self.statistics["backtracksolver.backtracks.nb"] += 1
        self.problem.domains = self.choice_points.pop()
        return True
