import numpy as np
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator


class AlldifferentPuget(Propagator):
    """
    JF Puget's bound consistency algorithm for the alldifferent constraint.
    """

    def compute_domains(self, domains: NDArray) -> NDArray:
        new_domains = np.zeros((self.variables.size, 2), dtype=int)
        return new_domains

    def __str__(self) -> str:
        return f"alldifferent({self.variables})"
