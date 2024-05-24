import numpy as np
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator


class Shift(Propagator):
    """
    Propagator for constraint x_i0 = x_i1 + c_i.
    """

    def __init__(self, variables: NDArray, constants: NDArray):
        super().__init__(variables)
        self.constants = constants.reshape(variables.shape[0], 1)

    def compute_domains(self, domains: NDArray) -> NDArray:
        new_domains = np.zeros((self.variables.size, 2), dtype=int)
        new_domains[self.variables[:, 0]] = domains[self.variables[:, 1]] + self.constants
        new_domains[self.variables[:, 1]] = domains[self.variables[:, 0]] - self.constants
        return new_domains

    def __str__(self) -> str:
        return f"{self.variables[:, 0]}={self.variables[:, 1]}+{self.constants.flatten()}"
