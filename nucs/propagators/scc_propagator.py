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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_scc(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 2 * n * n


def get_triggers_scc(n: int, parameters: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.full(n, dtype=np.uint8, fill_value=EVENT_MASK_MIN_MAX)


@njit(cache=False)  # Numba issue with cached recursive functions
def dfs_row(n: int, graph: NDArray, i: int, visited: NDArray) -> None:
    visited[i] = True
    for j in range(n):
        if graph[i, j] and not visited[j]:
            dfs_row(n, graph, j, visited)


@njit(cache=False)  # Numba issue with cached recursive functions
def dfs_col(n: int, graph: NDArray, j: int, visited: NDArray) -> None:
    visited[j] = True
    for i in range(n):
        if graph[i, j] and not visited[i]:
            dfs_col(n, graph, i, visited)


@njit(cache=False)  # Numba issue with recursive functions
def compute_domains_scc(domains: NDArray, parameters: NDArray) -> int:
    """
    :param domains: the domains of the variables
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    n = len(domains)
    graph = np.zeros((n, n), dtype=np.bool)
    for i in range(n):
        graph[i, domains[i, MIN] : (domains[i, MAX] + 1)] = True
    visited = np.zeros(n, dtype=np.bool)
    dfs_row(n, graph, 0, visited)
    if not np.all(visited):
        return PROP_INCONSISTENCY
    visited[:] = False
    dfs_col(n, graph, 0, visited)
    if not np.all(visited):
        return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
