from typing import List

from numpy.typing import NDArray

from ncs.problems.problem import Problem


class Heuristic:
    """
    Makes a choice.
    """

    def choose(self, changes: NDArray, choice_points: List[NDArray], problem: Problem) -> bool:  # type: ignore
        """
        Makes a choice.
        :return: True iff it is possible to make a choice
        """
        pass
