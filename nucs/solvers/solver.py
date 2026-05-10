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
from abc import abstractmethod
from typing import Callable, Dict, Iterator, List, Optional

from numba import njit  # type: ignore
from numpy.typing import NDArray
from rich import print

from nucs.constants import LOG_FORMAT, LOG_LEVEL_INFO, MIN
from nucs.problems.problem import Problem

logger = logging.getLogger(__name__)


class Solver:
    """
    A solver.
    """

    def __init__(self, problem: Optional[Problem], log_level: str = LOG_LEVEL_INFO):
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
    def get_statistics_as_dictionary(self) -> Dict[str, int]:
        """
        Returns the statistics as a dictionary.

        :return: a dictionary
        :rtype: Dict[str, int]
        """
        ...

    def print_statistics(self) -> None:
        """
        Pretty prints the statistics.
        """
        print(self.get_statistics_as_dictionary())

    @abstractmethod
    def solve(self) -> Iterator[NDArray]:
        """
        Returns an iterator over the solutions.

        :return: an iterator
        :rtype: Iterator[NDArray]
        """
        ...

    def solve_all(self, func: Optional[Callable] = None) -> None:
        """
        Finds all solutions.

        :param func: a function to handle each solution found
        :type func: Optional[Callable]
        """
        logger.info("Iterating over the solutions")
        for solution in self.solve():
            if func is not None:
                func(solution)

    def find_all(self) -> List[NDArray]:
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
    def minimize(self, variable_idx: int, mode: str) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that minimizes a given variable.

        :param variable_idx: the index of the variable
        :type variable_idx: int
        :param mode: the optimization mode
        :type mode: str

        :return: the solution if it exists or None
        :rtype: Optional[NDArray]
        """
        ...

    @abstractmethod
    def maximize(self, variable_idx: int, mode: str) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that maximizes a given variable.

        :param variable_idx: the index of the variable
        :type variable_idx: int
        :param mode: the optimization mode
        :type mode: str

        :return: the solution if it exists or None
        :rtype: Optional[NDArray]
        """
        ...


@njit(cache=True, fastmath=True)
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
