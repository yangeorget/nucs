from numpy.typing import NDArray

from ncs.problem import Problem


class ValueHeuristic:
    def make_value_choice(self, problem: Problem, idx: int) -> NDArray:  # type: ignore
        pass
