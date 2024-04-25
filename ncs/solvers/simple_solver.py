from typing import Iterator, Optional

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
        while True:
            solution = self.solveOne()
            if solution is None:
                break
            yield solution
            if not self.backtrack():
                break

    def solveOne(self) -> Optional[NDArray]:
        if not self.problem.filter():
            return None
        while not self.problem.is_solved():
            self.choice()  # let's make a choice
            while not self.problem.filter():  # the choice was not consistent
                if not self.backtrack():
                    return None
        return self.problem.domains


    def backtrack(self) -> bool:
        """
        Backtracks and updates the problem's domains
        :return: true iff it is possible to backtrack
        """
        # print("backtrack()")
        if len(self.choice_points) == 0:
            return False
        self.problem.domains = self.choice_points.pop()
        return True

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
