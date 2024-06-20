from abc import abstractmethod
from typing import List, Optional

from numpy.typing import NDArray

from ncs.problems.problem import Problem


class ValueHeuristic:
    """
    Chooses one or several values for a domain.
    """

    @abstractmethod
    def choose(self, choice_points: List[NDArray], problem: Problem, var_idx: int) -> Optional[NDArray]:
        """
        Chooses one or several values for a domain.
        :param choice_points: the choice point list
        :param problem: the problem
        :param var_idx: the index of the variable
        :return: the boolean array of shared domain changes or None if no choice can be made
        """
        pass
