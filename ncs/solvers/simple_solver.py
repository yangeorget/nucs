from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ncs.problem import MAX, MIN, Problem
from ncs.solvers.solver import Solver


class SimpleSolver(Solver):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.choice_points = []  # type: ignore

    def solve(self) -> Iterator[NDArray]:
        # print("solve()")
        if self.problem.filter():
            while not self.problem.is_solved():
                self.choice()  # let's make a choice
                if not self.problem.filter():  # the choice was not consistent
                    if not self.backtrack():
                        return
                yield self.problem.domains
                if not self.backtrack():
                    return
        # inconsistent problem

    def backtrack(self) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        # print("backtrack()")
        while len(self.choice_points) > 0:
            domains = self.choice_points.pop()
            self.problem.domains = domains
            if self.problem.filter():
                return True
        return False

    def choice(self) -> int:
        """
        Makes a choice by instantiating the first non instantiated variable to its first value
        :return: the index of the variable
        """
        # print("choice()")
        for idx in range(self.problem.domains.shape[0]):
            if not self.problem.is_instantiated(idx):
                domains = np.copy(self.problem.domains)
                self.problem.domains[idx, MAX] = domains[idx, MIN]
                domains[idx, MIN] += 1
                self.choice_points.append(domains)
                return idx
        return -1  # all variables are instantiated
