from abc import abstractmethod
from typing import List, Optional

from numpy.typing import NDArray

from ncs.problems.problem import Problem


class Heuristic:
    """
    Makes a choice.
    A choice can be more complex than just instantiating a variable.
    """

    @abstractmethod
    def choose(self, choice_points: List[NDArray], problem: Problem) -> Optional[NDArray]:
        """
        Makes a choice.
        :param choice_points: the choice points
        :param problem: the problem
        :return: the boolean array of shared domain changes or None if no choice can be made
        """
        pass
