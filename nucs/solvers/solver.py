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
from nucs.problems.problem import Problem


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
        """
        for solution in self.solve():
            if func is not None:
                func(solution)

    def find_all(self) -> List[NDArray]:
        """
        Finds all solutions.
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


def get_min_value(problem: Problem, var_idx: int) -> int:
    """
    Gets the minimal value of a variable.
    :param var_idx: the index of the variable
    :return: the minimal value
    """
    return problem.shr_domains_arr[problem.dom_indices_arr[var_idx], MIN] + problem.dom_offsets_arr[var_idx]


def get_max_value(problem: Problem, var_idx: int) -> int:
    """
    Gets the maximal value of a variable.
    :param var_idx: the index of the variable
    :return: the maximal value
    """
    return problem.shr_domains_arr[problem.dom_indices_arr[var_idx], MAX] + problem.dom_offsets_arr[var_idx]


def set_min_value(problem: Problem, var_idx: int, min_value: int) -> None:
    """
    Sets the minimal value of a variable.
    :param var_idx: the index of the variable
    :param min_value: the minimal value
    """
    problem.shr_domains_arr[problem.dom_indices_arr[var_idx], MIN] = min_value - problem.dom_offsets_arr[var_idx]


def set_max_value(problem: Problem, var_idx: int, max_value: int) -> None:
    """
    Sets the maximal value of a variable.
    :param var_idx: the index of the variable
    :param min_value: the maximal value
    """
    problem.shr_domains_arr[problem.dom_indices_arr[var_idx], MAX] = max_value - problem.dom_offsets_arr[var_idx]


def get_solution(problem: Problem) -> NDArray:
    """
    Returns the solution to the problem.
    :return: a Numpy array
    """
    return problem.shr_domains_arr[problem.dom_indices_arr, MIN] + problem.dom_offsets_arr


@njit(cache=True)
def is_solved(shr_domains: NDArray) -> bool:
    """
    Returns true iff the problem is solved.
    :return: a boolean
    """
    return bool(np.all(np.equal(shr_domains[:, MIN], shr_domains[:, MAX])))


def decrease_max(problem: Problem, var_idx: int, value: int) -> None:
    """
    Decreases the max of a variable
    :param problem: the problem
    :param var_idx: the index of the variable
    :param value: the current max
    """
    set_max_value(problem, var_idx, value - 1)


def increase_min(problem: Problem, var_idx: int, value: int) -> None:
    """
    Increases the min of a variable
    :param problem: the problem
    :param var_idx: the index of the variable
    :param value: the current min
    """
    set_min_value(problem, var_idx, value + 1)
