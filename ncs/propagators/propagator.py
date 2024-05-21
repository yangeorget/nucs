from numpy.typing import NDArray


class Propagator:
    """
    Abstraction for a bound-consistency algorithm.
    """

    def __init__(self, variables: NDArray):
        self.variables = variables

    def compute_domains(self, domains: NDArray) -> NDArray:  # type: ignore
        """
        Computes new domains for the variables
        :param domains: the initial domains
        :return: the new domains
        """
        pass
