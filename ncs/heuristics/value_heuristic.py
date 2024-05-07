from numpy.typing import NDArray

from ncs.problem import Problem


class ValueHeuristic:
    """
    Chooses a value.
    """

    def make_value_choice(self, problem: Problem, idx: int) -> NDArray:  # type: ignore
        pass
