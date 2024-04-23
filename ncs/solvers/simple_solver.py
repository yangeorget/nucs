from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ncs.problem import MAX, MIN, Problem
from ncs.solvers.solver import Solver


class SimpleSolver(Solver):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.choice_points = []  # type: ignore

    def solve(self) -> Optional[NDArray]:
        print("solve()")
        if self.problem.filter():
            while self.choice() >= 0:
                # a choice could be made
                while not self.problem.filter():
                    # the choice was not consistent
                    if not self.backtrack():
                        return None
            if self.problem.is_solved():
                return self.problem.domains
        return None

    def backtrack(self) -> bool:
        print("backtrack()")
        if len(self.choice_points) == 0:
            return False
        domains = self.choice_points.pop()
        self.problem.domains = domains
        return True

    def choice(self) -> int:
        print("choice()")
        for idx in range(self.problem.domains.shape[0]):
            if not self.problem.is_instantiated(idx):
                domains = np.copy(self.problem.domains)
                self.problem.domains[idx, MAX] = domains[idx, MIN]
                domains[idx, MIN] += 1
                self.choice_points.append(domains)
                return idx
        return -1  # all variables are instantiated
