from typing import Iterator, List

from ncs.problems.problem import Problem


class Solver:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.statistics = {"solver.solutions.nb": 0, "problem.filters.nb": 0, "problem.propagators.filters.nb": 0}

    def solve(self) -> Iterator[List[int]]:  # type: ignore
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        pass

    def solve_all(self) -> List[List[int]]:
        return [s for s in self.solve()]
