from typing import List

from numpy._typing import NDArray

from ncs.problem import Problem


class Heuristic:
    def makeChoice(self, choice_points: List[NDArray], problem: Problem) -> bool:  # type: ignore
        """
        Makes a choice.
        :return: True iff it is possible to make a choice
        """
        pass
