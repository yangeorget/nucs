from numpy.typing import NDArray

from ncs.problem import Problem


class ValueHeuristic:
    """
    Chooses a value for a domain.
    """

    def make_value_choice(self, problem: Problem, idx: int) -> NDArray:  # type: ignore
        """
        Chooses a value for a domain.
        :param problem: the problem
        :param idx: the index of the variable
        :return: the new domains
        """
        pass
