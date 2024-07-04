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
        pass
