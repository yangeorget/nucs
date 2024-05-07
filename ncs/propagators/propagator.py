import numpy as np
from numpy.typing import NDArray


class Propagator:
    """
    Abstraction for a bound-consistency algorithm.
    """

    def __init__(self, variables: NDArray):
        self.variables = variables
        self.mask = np.ones((len(self.variables), 2), dtype=bool)

    def compute_domains(self, domains: NDArray) -> NDArray:  # type: ignore
        """
        Computes new domains for the variables
        :param domains: the initial domains
        :return: the new domains
        """
        pass
