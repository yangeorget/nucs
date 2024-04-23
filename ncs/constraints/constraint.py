from numpy.typing import NDArray


class Constraint:
    def __init__(self, variables: NDArray):
        self.variables = variables

    def compute_domains(self, domains: NDArray) -> NDArray:  # type: ignore
        pass
