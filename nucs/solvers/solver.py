###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024-2026 - Yan Georget
###############################################################################
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from numba import njit  # type: ignore
from numpy.typing import NDArray
from rich import print

from nucs.constants import DOMAIN_MIN, LOG_FORMAT, LOG_LEVEL_INFO, OPTIM_RESET
from nucs.problems.problem import Problem

logger = logging.getLogger(__name__)


class Solver(ABC):
    """
    A solver.
    """

    def __init__(self, problem: Problem | None, log_level: str = LOG_LEVEL_INFO):
        """
        Initializes the solver.

        :param problem: a problem or None
        :type problem: Optional[Problem]
        :param log_level: the log level as a string
        :type log_level: str
        """
        logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, log_level))
        logging.getLogger("numba").setLevel(logging.WARNING)
        logger.info("Initializing Solver")
        self.timed_out = False
        if problem is not None:
            self.problem = problem
            problem.init()

    @abstractmethod
    def solve(self, timeout: float | None = None) -> Iterator[NDArray]:
        """
        Returns an iterator over the solutions.

        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]

        :return: an iterator
        :rtype: Iterator[NDArray]
        """
        ...

    def solve_all(self, func: Callable | None = None, timeout: float | None = None) -> None:
        """
        Finds all solutions.

        :param func: a function to handle each solution found
        :type func: Optional[Callable]
        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]
        """
        logger.info("Iterating over the solutions")
        for solution in self.solve(timeout):
            if func is not None:
                func(solution)

    def find_all(self, timeout: float | None = None) -> list[NDArray]:
        """
        Finds all solutions.

        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]

        :return: the list of all solutions
        :rtype: List[NDArray]
        """
        logger.info("Returning all solutions")
        solutions = []
        self.solve_all(lambda solution: solutions.append(solution), timeout)
        return solutions

    @abstractmethod
    def optimize(self, variable: int, bound: int, mode: str, timeout: float | None = None) -> Iterator[NDArray]:
        """
        Iterates over the successively improving solutions found while optimizing a given variable.

        Each yielded solution improves on the previous one; the last yielded solution is the optimum.
        Nothing is yielded when the problem is unsatisfiable. Under a timeout the last yielded solution
        is the best one found within the budget, which is not necessarily optimal: check :attr:`timed_out`.

        :param variable: the variable
        :type variable: int
        :param bound: MIN to minimize the variable, MAX to maximize it
        :type bound: int
        :param mode: the optimization mode
        :type mode: str
        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]

        :return: an iterator over the improving solutions, the last one being optimal
        :rtype: Iterator[NDArray]
        """
        ...

    def optimize_all(
        self,
        variable: int,
        bound: int,
        mode: str = OPTIM_RESET,
        func: Callable | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Finds all the successively improving solutions while optimizing a given variable.

        :param variable: the variable
        :type variable: int
        :param bound: MIN to minimize the variable, MAX to maximize it
        :type bound: int
        :param mode: the optimization mode
        :type mode: str
        :param func: a function to handle each improving solution found
        :type func: Optional[Callable]
        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]
        """
        logger.info("Iterating over the solutions")
        for solution in self.optimize(variable, bound, mode, timeout):
            if func is not None:
                func(solution)

    def find_best(
        self, variable: int, bound: int, mode: str = OPTIM_RESET, timeout: float | None = None
    ) -> NDArray | None:
        """
        Finds, if it exists, the solution to the problem that optimizes a given variable.

        Under a timeout the returned solution is the best one found within the budget, which is not
        necessarily optimal: check :attr:`timed_out`.

        :param variable: the variable
        :type variable: int
        :param bound: MIN to minimize the variable, MAX to maximize it
        :type bound: int
        :param mode: the optimization mode
        :type mode: str
        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]

        :return: the solution if it exists or None
        :rtype: Optional[NDArray]
        """
        logger.info("Returning the optimal solution")
        best_solution = None
        for solution in self.optimize(variable, bound, mode, timeout):
            best_solution = solution
        return best_solution

    def _expired(self, deadline: float | None) -> bool:
        """
        Returns whether the search budget is spent, recording the fact in :attr:`timed_out`.

        The check can only run where the search returns to Python -- between two solutions -- because a
        single descent runs in compiled code that nothing in this process can interrupt. It therefore bounds
        the time spent *between* solutions, not the time spent inside one descent.

        :param deadline: the monotonic time to stop at, or None for an unbounded search
        :type deadline: Optional[float]

        :return: True when the deadline has passed
        :rtype: bool
        """
        if deadline is None or time.monotonic() < deadline:
            return False
        logger.info("Timeout reached, stopping the search")
        self.timed_out = True
        return True

    @abstractmethod
    def get_statistics_as_dictionary(self) -> dict[str, int]:
        """
        Returns the statistics as a dictionary.

        :return: a dictionary mapping statistic labels to values
        :rtype: Dict[str, int]
        """
        ...

    def print_statistics(self) -> None:
        """
        Pretty prints the statistics.
        """
        print(self.get_statistics_as_dictionary())


@njit(cache=True)
def get_solution(domains: NDArray) -> NDArray:
    """
    Returns the solution to the problem.

    :param domains: the domains, every one of them ground
    :type domains: NDArray

    :return: a Numpy array
    :rtype: NDArray
    """
    return domains[:, DOMAIN_MIN].copy()
