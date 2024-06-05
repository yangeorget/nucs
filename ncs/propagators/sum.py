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
        new_domains = np.zeros((self.size, 2), dtype=int)
        new_domains[0] = np.sum(domains[1:], axis=0)
        new_domains[1:, MIN] = domains[0, MIN] + domains[1:, MAX] - new_domains[0, MAX]
        new_domains[1:, MAX] = domains[0, MAX] + domains[1:, MIN] - new_domains[0, MIN]
        return new_domains
