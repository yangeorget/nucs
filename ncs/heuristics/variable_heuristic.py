from typing import List

from numpy.typing import NDArray

from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.value_heuristic import ValueHeuristic
from ncs.problem import Problem


class VariableHeuristic(Heuristic):
    def __init__(self, value_heuristic: ValueHeuristic):
        self.value_heuristic = value_heuristic

    def make_choice(self, choice_points: List[NDArray], problem: Problem) -> bool:
        idx = self.make_variable_choice(problem)
        if idx == -1:
            return False
        domains = self.value_heuristic.make_value_choice(problem, idx)
        choice_points.append(domains)
        return True

    def make_variable_choice(self, problem: Problem) -> int:  # type: ignore
        pass
