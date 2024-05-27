import numpy as np
from numpy.typing import NDArray

from ncs.propagators.propagator import Propagator


class AlldifferentPugetN3(Propagator):
    """
    JF Puget's bound consistency algorithm for the alldifferent constraint.
    """

    def compute_domains(self, domains: NDArray) -> NDArray:
        new_domains = np.zeros((self.variables.size, 2), dtype=int)
        return new_domains

    def __str__(self) -> str:
        return f"alldifferent_puget_n3({self.variables})"
