from typing import Iterator

from numpy.typing import NDArray

from ncs.problems.problem import Problem


class Solver:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.statistics = {"solver.solutions.nb": 0, "problem.filters.nb": 0}

    def solve(self) -> Iterator[NDArray]:  # type: ignore
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        pass
