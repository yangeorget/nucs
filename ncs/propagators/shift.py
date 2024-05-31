from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator


class Shift(Propagator):
    """
    Propagator for constraint x_i = x_i+n + c_i.
    """

    def __init__(self, variables: List[int], constants: List[int]):
        super().__init__(variables)
        self.n = self.size // 2
        self.constants = np.array(constants).reshape(self.n, 1)
        self.swapped_variables = np.concatenate((self.variables[self.n :], self.variables[: self.n]))
        self.all_constants = np.concatenate((self.constants, -self.constants))

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        return domains[self.swapped_variables] + self.all_constants

    def __str__(self) -> str:
        return f"{self.variables[0 : self.n]}={self.variables[self.n : self.size]}+{self.constants.flatten()}"
