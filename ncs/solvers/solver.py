from typing import Iterator, List

from ncs.problems.problem import Problem
from ncs.utils import stats_init


class Solver:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.statistics = stats_init()

    def solve(self) -> Iterator[List[int]]:  # type: ignore
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        pass

    def solve_all(self) -> List[List[int]]:
        return [s for s in self.solve()]
