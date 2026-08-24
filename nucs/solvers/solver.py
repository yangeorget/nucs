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
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from numba import njit  # type: ignore
from numpy.typing import NDArray
from rich import print

from nucs.constants import LOG_FORMAT, LOG_LEVEL_INFO, MIN, OPTIM_RESET
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
        if problem is not None:
            self.problem = problem
            problem.init()

    @abstractmethod
    def solve(self) -> Iterator[NDArray]:
        """
        Returns an iterator over the solutions.

        :return: an iterator
        :rtype: Iterator[NDArray]
        """
        ...

    def solve_all(self, func: Callable | None = None) -> None:
        """
        Finds all solutions.

        :param func: a function to handle each solution found
        :type func: Optional[Callable]
        """
        logger.info("Iterating over the solutions")
        for solution in self.solve():
            if func is not None:
                func(solution)

    def find_all(self) -> list[NDArray]:
        """
        Finds all solutions.

        :return: the list of all solutions
        :rtype: List[NDArray]
        """
        logger.info("Returning all solutions")
        solutions = []
        self.solve_all(lambda solution: solutions.append(solution))
        return solutions

    @abstractmethod
    def optimize(self, variable: int, bound: int, mode: str) -> Iterator[NDArray]:
        """
        Iterates over the successively improving solutions found while optimizing a given variable.

        Each yielded solution improves on the previous one; the last yielded solution is the optimum.
        Nothing is yielded when the problem is unsatisfiable. Consumers that only need the optimum should
        use :meth:`find_best`; streaming consumers (e.g. the FlatZinc runner) print each solution as it
        is produced.

        :param variable: the variable
        :type variable: int
        :param bound: MIN to minimize the variable, MAX to maximize it
        :type bound: int
        :param mode: the optimization mode
        :type mode: str

        :return: an iterator over the improving solutions, the last one being optimal
        :rtype: Iterator[NDArray]
        """
        ...

    def optimize_all(self, variable: int, bound: int, mode: str = OPTIM_RESET, func: Callable | None = None) -> None:
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
        """
        logger.info("Iterating over the solutions")
        for solution in self.optimize(variable, bound, mode):
            if func is not None:
                func(solution)

    def find_best(self, variable: int, bound: int, mode: str = OPTIM_RESET) -> NDArray | None:
        """
        Finds, if it exists, the solution to the problem that optimizes a given variable.

        :param variable: the variable
        :type variable: int
        :param bound: MIN to minimize the variable, MAX to maximize it
        :type bound: int
        :param mode: the optimization mode
        :type mode: str

        :return: the solution if it exists or None
        :rtype: Optional[NDArray]
        """
        logger.info("Returning the optimal solution")
        best_solution = None
        for solution in self.optimize(variable, bound, mode):
            best_solution = solution
        return best_solution

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
def get_solution(domains_stk: NDArray, top: int) -> NDArray:
    """
    Returns the solution to the problem.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param top: the index of the top of the stacks
    :type top: int

    :return: a Numpy array
    :rtype: NDArray
    """
    return domains_stk[top, :, MIN].copy()
