from abc import abstractmethod

from numpy.typing import NDArray


class Heuristic:
    """
    Makes a choice.
    A choice can be more complex than just instantiating a variable.
    """

    @abstractmethod
    def choose(self, shr_domains: NDArray, shr_domain_changes: NDArray, dom_indices: NDArray) -> NDArray:
        """
        Makes a choice.
        :param shr_domains: the shared domains of the problem
        :param dom_indices: the domain indices of the problem variables
        :return: the new shared domain to be added to the choice point
        """
        pass
