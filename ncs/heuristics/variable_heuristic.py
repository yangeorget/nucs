from typing import List, Optional

from numpy.typing import NDArray

from ncs.heuristics.heuristic import Heuristic
from ncs.heuristics.value_heuristic import ValueHeuristic
from ncs.problems.problem import Problem


class VariableHeuristic(Heuristic):
    """
    Chooses a variable and one or several values for this variable.
    """

    def __init__(self, value_heuristic: ValueHeuristic):
        self.value_heuristic = value_heuristic

    def choose(self, choice_points: List[NDArray], problem: Problem) -> Optional[NDArray]:
        var_idx = self.choose_variable(problem)
        if var_idx == -1:
            return None
        return self.value_heuristic.choose(choice_points, problem, var_idx)

    def choose_variable(self, problem: Problem) -> int:  # type: ignore
        """
        Chooses a variable.
        :param problem: a problem
        :return: the index of the variable
        """
        pass
