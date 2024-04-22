import numpy as np
from numpy.typing import NDArray

from ncs.constraints.constraint import Constraint
from ncs.problem import MAX, MIN, Problem


class Sum(Constraint):
    def compute_domains(self, problem: Problem) -> NDArray:
        print(f"compute_domains{problem})")
        domains = np.full((len(self.variables), 2), 0)
        x = self.variables[0]
        y = self.variables[1:]
        domains[0] = np.sum(problem.domains[y], axis=0)
        domains[1:, MIN] = problem.domains[x, MIN] + problem.domains[y, MAX] - domains[0, MAX]
        domains[1:, MAX] = problem.domains[x, MAX] + problem.domains[y, MIN] - domains[0, MIN]
        return domains

    def __str__(self) -> str:
        return f"{self.variables[0]}=sum({self.variables[1:]})"
