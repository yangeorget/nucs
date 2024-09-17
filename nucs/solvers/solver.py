from typing import Iterator, List, Optional

from nucs.problems.problem import Problem


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

    def find_all(self) -> None:
        """
        Finds all solutions.
        """
        for _ in self.solve():
            pass

    def minimize(self, var_idx: int) -> Optional[List[int]]:  # type: ignore
        """
        Finds, if it exists, the solution to the problem that minimizes a given variable.
        :param variable_idx: the index of the variable
        :return: the solution if it exists or None
        """
        pass
