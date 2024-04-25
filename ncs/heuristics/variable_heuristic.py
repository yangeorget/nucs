from typing import List

from numpy.typing import NDArray

from ncs.heuristics.heuristic import Heuristic
from ncs.problem import Problem


class VariableHeuristic(Heuristic):
    def makeChoice(self, choice_points: List[NDArray], problem: Problem) -> bool:
        # print("makeChoice()")
        for idx in range(problem.domains.shape[0]):
            if not problem.is_instantiated(idx):
                domains = self.makeVariableChoice(problem, idx)
                choice_points.append(domains)
                return True
        return False

    def makeVariableChoice(self, problem: Problem, idx: int) -> NDArray:  # type: ignore
        pass
