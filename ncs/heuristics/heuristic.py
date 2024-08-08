from abc import abstractmethod
from typing import Tuple

from numpy.typing import NDArray


class Heuristic:
    """
    Makes a choice.
    A choice can be more complex than just instantiating a variable.
    """

    @abstractmethod
    def choose(self, shr_domains: NDArray, dom_indices: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Makes a choice.
        :param shr_domains: the shared domains of the problem
        :param dom_indices: the domain indices of the problem variables
        :return: the new shared domain to be added to the choice point and the changes to the actual domains
        """
        pass
