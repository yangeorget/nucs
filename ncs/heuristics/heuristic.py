from typing import List

from numpy.typing import NDArray

from ncs.problem import Problem


class Heuristic:
    """
    Makes a choice.
    """

    def make_choice(self, choice_points: List[NDArray], problem: Problem) -> bool:  # type: ignore
        """
        Makes a choice.
        :return: True iff it is possible to make a choice
        """
        pass
