from typing import Callable, Iterator, List, Optional

from nucs.problems.problem import Problem
from nucs.statistics import init_statistics


class Solver:
    """
    A solver.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.statistics = init_statistics()

    def solve(self) -> Iterator[List[int]]:  # type: ignore
        """
        Returns an iterator over the solutions.
        :return: an iterator
        """
        pass

    def solve_all(self, func: Optional[Callable] = None) -> None:
        """
        Finds all solutions.
        """
        for solution in self.solve():
            if func is not None:
                func(solution)

    def find_all(self) -> List[List[int]]:
        """
        Finds all solutions.
        """
        solutions = []
        self.solve_all(lambda solution: solutions.append(solution))
        return solutions

    def minimize(self, var_idx: int) -> Optional[List[int]]:  # type: ignore
        """
        Finds, if it exists, the solution to the problem that minimizes a given variable.
        :param variable_idx: the index of the variable
        :return: the solution if it exists or None
        """
        pass

    def maximize(self, var_idx: int) -> Optional[List[int]]:  # type: ignore
        """
        Finds, if it exists, the solution to the problem that maximizes a given variable.
        :param variable_idx: the index of the variable
        :return: the solution if it exists or None
        """
        pass
