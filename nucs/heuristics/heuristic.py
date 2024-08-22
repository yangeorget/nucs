from abc import abstractmethod
from typing import List

from numpy.typing import NDArray


class Heuristic:
    """
    Makes a choice.
    A choice can be more complex than just instantiating a variable.
    """

    @abstractmethod
    def choose(
        self, choice_points: List[NDArray], shr_domains: NDArray, shr_domain_changes: NDArray, dom_indices: NDArray
    ) -> None:
        """
        Makes a choice.
        :param choice_points: the choice point stack
        :param shr_domains: the shared domains of the problem
        :param shr_domain_changes: an array of shared domain changes
        :param dom_indices: the domain indices of the problem variables
        """
        pass
