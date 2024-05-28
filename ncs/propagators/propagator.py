from typing import Optional

from numpy.typing import NDArray


class Propagator:
    """
    Abstraction for a bound-consistency algorithm.
    """

    def __init__(self, variables: NDArray):
        self.variables = variables

    def compute_domains(self, domains: NDArray) -> Optional[NDArray]:
        """
        Computes new domains for the variables
        :param domains: the initial domains
        :return: the new domains or None if there is an inconsistency
        """
        pass
