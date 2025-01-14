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

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import LOG_FORMAT, LOG_LEVEL_INFO, MAX, MIN
from nucs.problems.problem import Problem

logger = logging.getLogger(__name__)


class Solver:
    """
    A solver.
    """

    def __init__(self, problem: Optional[Problem], log_level: str = LOG_LEVEL_INFO):
        """
        Inits the solver.
        :param problem: a problem or None
        :param log_level: the log level as a string
        """
        logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, log_level))  # TODO: move to examples
        logger.debug("Initializing Solver")
        if problem is not None:
            self.problem = problem
            problem.init()

    @abstractmethod
    def get_statistics(self) -> Dict[str, int]:
        """
        Returns the statistics as a dictionary.
        :return: a dictionary
        """
        ...

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
        :return: the solution if it exists or None
        """
        ...

    @abstractmethod
    def maximize(self, variable_idx: int, mode: str) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that maximizes a given variable.
        :param variable_idx: the index of the variable
        :return: the solution if it exists or None
        """
        ...


@njit(cache=True)
def get_solution(
    shr_domains_stack: NDArray, stacks_top: NDArray, dom_indices_arr: NDArray, dom_offsets_arr: NDArray
) -> NDArray:
    """
    Returns the solution to the problem.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_indices_arr: the domain indices
    :param dom_offsets_arr: the domain offsets
    :return: a Numpy array
    """
    return shr_domains_stack[stacks_top[0], dom_indices_arr, MIN] + dom_offsets_arr


@njit(cache=True)
def is_solved(shr_domains_stack: NDArray, stacks_top: NDArray) -> bool:
    """
    Returns true iff the problem is solved.
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: a boolean
    """
    return bool(np.all(np.equal(shr_domains_stack[stacks_top[0], :, MIN], shr_domains_stack[stacks_top[0], :, MAX])))
