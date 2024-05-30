from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problems.problem import MAX, MIN
from ncs.propagators.propagator import Propagator


class Sum(Propagator):
    """
    Propagator for constraint x_0 = sum(x_i) where i > 0.
    """

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        new_domains = self.init_domains()
        x = self.variables[0]
        y = self.variables[1:]
        new_domains[0] = np.sum(domains[y], axis=0)
        new_domains[1:, MIN] = domains[x, MIN] + domains[y, MAX] - new_domains[0, MAX]
        new_domains[1:, MAX] = domains[x, MAX] + domains[y, MIN] - new_domains[0, MIN]
        return new_domains

    def __str__(self) -> str:
        return f"{self.variables[0]}=sum({self.variables[1:]})"
