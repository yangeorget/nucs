from abc import abstractmethod
from typing import Tuple

from numpy.typing import NDArray


class Heuristic:
    """
    Makes a choice.
    A choice can be more complex than just instantiating a variable.
    """

    @abstractmethod
    def choose(self, shared_domains: NDArray, domain_indices: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Makes a choice.
        :param shared_domains: the shared domains of the problem
        :param domain_indices: the domain indices of the problem variables
        :return: the new shared domain to be added to the choice point and the changes to the actual domains
        """
        pass
