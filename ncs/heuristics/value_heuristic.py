from numpy.typing import NDArray

from ncs.problem import Problem


class ValueHeuristic:
    """
    Chooses one or several values for a domain.
    """

    def choose(self, problem: Problem, idx: int) -> NDArray:  # type: ignore
        """
        Chooses one or several values for a domain.
        :param problem: the problem
        :param idx: the index of the variable
        :return: the new domains
        """
        pass
