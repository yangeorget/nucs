from typing import Iterator, List, Optional

from ncs.problems.problem import Problem


class Solver:
    """
    A solver.
    """

    def __init__(self, problem: Problem):
        self.problem = problem

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
        """
        Returns the solution that minimizes the given variable.
        :param var_idx: the index of the variable to minimize
        :return: the solution that minimizes the variable if it exists or None
        """
        pass
