from abc import abstractmethod
from typing import List, Optional

from numpy.typing import NDArray

from ncs.problems.problem import Problem


class Heuristic:
    """
    Makes a choice.
    """

    @abstractmethod
    def choose(self, choice_points: List[NDArray], problem: Problem) -> Optional[NDArray]:
        """
        Makes a choice.
        """
        pass
