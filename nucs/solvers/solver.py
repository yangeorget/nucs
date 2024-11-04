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
# Copyright 2024 - Yan Georget
###############################################################################
from abc import abstractmethod
from typing import Callable, Iterator, List, Optional

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


class Solver:
    """
    A solver.
    """

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
        solutions = []
        self.solve_all(lambda solution: solutions.append(solution))
        return solutions

    @abstractmethod
    def minimize(self, variable_idx: int) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that minimizes a given variable.
        :param variable_idx: the index of the variable
        :return: the solution if it exists or None
        """
        ...

    @abstractmethod
    def maximize(self, variable_idx: int) -> Optional[NDArray]:
        """
        Finds, if it exists, the solution to the problem that maximizes a given variable.
        :param variable_idx: the index of the variable
        :return: the solution if it exists or None
        """
        ...


@njit(cache=True)
def get_solution(shr_domains_arr: NDArray, dom_indices_arr: NDArray, dom_offsets_arr: NDArray) -> NDArray:
    """
    Returns the solution to the problem.
    :return: a Numpy array
    """
    return shr_domains_arr[dom_indices_arr, MIN] + dom_offsets_arr


@njit(cache=True)
def is_solved(shr_domains: NDArray) -> bool:
    """
    Returns true iff the problem is solved.
    :return: a boolean
    """
    return bool(np.all(np.equal(shr_domains[:, MIN], shr_domains[:, MAX])))


@njit(cache=True)
def decrease_max(
    shr_domains_arr: NDArray, dom_indices_arr: NDArray, dom_offsets_arr: NDArray, var_idx: int, value: int
) -> None:
    """
    Decreases the max of a variable
    :param var_idx: the index of the variable
    :param value: the current max
    """
    shr_domains_arr[dom_indices_arr[var_idx], MAX] = value - 1 - dom_offsets_arr[var_idx]


@njit(cache=True)
def increase_min(
    shr_domains_arr: NDArray, dom_indices_arr: NDArray, dom_offsets_arr: NDArray, var_idx: int, value: int
) -> None:
    """
    Increases the min of a variable
    :param var_idx: the index of the variable
    :param value: the current min
    """
    shr_domains_arr[dom_indices_arr[var_idx], MIN] = value + 1 - dom_offsets_arr[var_idx]
