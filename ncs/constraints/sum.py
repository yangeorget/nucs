import numpy as np
from numpy.typing import NDArray

from ncs.constraints.constraint import Constraint


class Sum(Constraint):
    def __init__(self, variables: NDArray):
        super().__init__(variables)
        self.x = variables[0]
        self.y = variables[1:]

    def compute_domains(self) -> NDArray:
        domains = np.full((len(self.variables), 2), 0)
        domains[self.x] = np.sum(self.problem.domains[self.y], axis=0)
        domains[self.y, 0] = self.problem.domains[self.x, 0] + self.problem.domains[self.y, 1] - domains[self.x, 1]
        domains[self.y, 1] = self.problem.domains[self.x, 1] + self.problem.domains[self.y, 0] - domains[self.x, 0]
        return domains
