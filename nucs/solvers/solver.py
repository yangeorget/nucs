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
# Copyright 2024-2025 - Yan Georget
###############################################################################
import logging
from abc import abstractmethod
from typing import Callable, Dict, Iterator, List, Optional

from numba import njit  # type: ignore
from numpy.typing import NDArray
from rich import print

from nucs.constants import LOG_FORMAT, LOG_LEVEL_INFO, MAX, MIN
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
        :param log_level: the log level as a string
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
        """
        ...

    def solve_all(self, func: Optional[Callable] = None) -> None:
        """
        Finds all solutions.
        :param func: a function to handle each solution found
        """
        for solution in self.solve():
            if func is not None:
                func(solution)

    def find_all(self) -> List[NDArray]:
        """
        Finds all solutions.
        :return: the list of all solutions
        """
        logger.info("Finding all solutions")
        solutions = []
        self.solve_all(lambda solution: solutions.append(solution))
        return solutions

    def find_one(self) -> Optional[NDArray]:
        logger.info("Finding one solution")
        for solution in self.solve():
            return solution
        return None

    @abstractmethod
    def minimize(self, variable_idx: int, mode: str) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that minimizes a given variable.
        :param variable_idx: the index of the variable
        :param mode: the optimization mode
        :return: the solution if it exists or None
        """
        ...

    @abstractmethod
    def maximize(self, variable_idx: int, mode: str) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that maximizes a given variable.
        :param variable_idx: the index of the variable
        :param mode: the optimization mode
        :return: the solution if it exists or None
        """
        ...


@njit(cache=True)
def get_solution(domains_stk: NDArray, stks_top: NDArray) -> NDArray:
    """
    Returns the solution to the problem.
    :param domains_stk: the stack of domains
    :param stks_top: the index of the top of the stacks as a Numpy array
    :return: a Numpy array
    """
    return domains_stk[stks_top[0], :, MIN].copy()


@njit(cache=True)
def is_solved(domains_stk: NDArray, top: int) -> bool:
    """
    Returns true iff the problem is solved.
    :param domains_stk: the stack of domains
    :param top: the index of the top of the stacks
    :return: a boolean
    """
    for var in range(domains_stk.shape[1]):
        if domains_stk[top, var, MIN] != domains_stk[top, var, MAX]:
            return False
    return True
