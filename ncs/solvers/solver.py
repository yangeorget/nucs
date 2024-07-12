from typing import Iterator, List, Optional

from ncs.problems.problem import Problem
from ncs.utils import statistics_init


class Solver:
    """
    A solver.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.statistics = statistics_init()

    def solve(self) -> Iterator[List[int]]:  # type: ignore
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        pass

    def solve_all(self) -> List[List[int]]:
        """
        Returns the list of all solutions.
        :return: a list of list of integers
        """
        return [s for s in self.solve()]

    def minimize(self, var_idx: int) -> Optional[List[int]]:  # type: ignore
        pass
