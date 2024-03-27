from typing import Any

from numpy.typing import NDArray


class Constraint:
    def __init__(self, variables: NDArray):
        self.variables = variables
        self.problem: Any = None

    def filter(self) -> NDArray:
        new_domains = self.compute_domains()
        return self.problem.update_domains(new_domains)

    def compute_domains(self) -> NDArray:  # type: ignore
        pass
