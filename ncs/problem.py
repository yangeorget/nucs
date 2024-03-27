from typing import List

import numpy as np
from numpy.typing import NDArray

from ncs.constraints.constraint import Constraint


class Problem:
    def __init__(self, domains: NDArray, constraints: List[Constraint] = []):
        self.domains = domains
        self.constraints = constraints

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)
        constraint.problem = self

    def update_domains(self, new_domains: NDArray) -> NDArray:
        changes = np.full((len(new_domains), 2), False)
        new_minimums = np.maximum(new_domains[:, 0], self.domains[:, 0])
        np.greater(new_minimums, self.domains[:, 0], out=changes[:, 0])
        self.domains[:, 0] = new_minimums
        new_maximums = np.minimum(new_domains[:, 1], self.domains[:, 1])
        np.less(new_maximums, self.domains[:, 1], out=changes[:, 1])
        self.domains[:, 1] = new_maximums
        return changes
