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

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_lexicographic_leq(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 4 * n


def get_triggers_lexicographic_leq(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return np.ones((n, 2), dtype=np.bool)


@njit(cache=True)
def compute_domains_4(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int) -> int:
    while i < n and x[i, MIN] == y[i, MAX]:
        i += 1
        # s = i
    if i < n and x[i, MIN] > y[i, MAX]:
        # xq < yq
        x[q, MAX] = min(x[q, MAX], y[q, MAX] - 1)
        if x[q, MAX] < x[q, MIN]:
            return PROP_INCONSISTENCY
        y[q, MIN] = max(y[q, MIN], x[q, MIN] + 1)
        if y[q, MAX] < y[q, MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, MAX] < y[q, MIN] else PROP_CONSISTENCY
    # u = 4
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_3(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int) -> int:
    while i < n and x[i, MAX] == y[i, MIN]:
        i += 1
        # s = i
    if i == n or x[i, MAX] < y[i, MIN]:
        # xq <= yq
        x[q, MAX] = min(x[q, MAX], y[q, MAX])
        if x[q, MAX] < x[q, MIN]:
            return PROP_INCONSISTENCY
        y[q, MIN] = max(y[q, MIN], x[q, MIN])
        if y[q, MAX] < y[q, MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, MAX] <= y[q, MIN] else PROP_CONSISTENCY
    # u = 3
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_2(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int) -> int:
    while i < n and x[i, MIN] == x[i, MAX] == y[i, MIN] == y[i, MAX]:
        i += 1
        r = i
    if i == n or x[i, MAX] < y[i, MIN]:
        # xq <= yq
        x[q, MAX] = min(x[q, MAX], y[q, MAX])
        if x[q, MAX] < x[q, MIN]:
            return PROP_INCONSISTENCY
        y[q, MIN] = max(y[q, MIN], x[q, MIN])
        if y[q, MAX] < y[q, MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, MAX] <= y[q, MIN] else PROP_CONSISTENCY
    if x[i, MIN] > y[i, MAX]:
        # xq < yq
        x[q, MAX] = min(x[q, MAX], y[q, MAX] - 1)
        if x[q, MAX] < x[q, MIN]:
            return PROP_INCONSISTENCY
        y[q, MIN] = max(y[q, MIN], x[q, MIN] + 1)
        if y[q, MAX] < y[q, MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, MAX] < y[q, MIN] else PROP_CONSISTENCY
    if x[i, MAX] == y[i, MIN] and x[i, MIN] < y[i, MAX]:
        if s > i + 1:
            i = s
        else:
            i += 1
            s = i
        return compute_domains_3(x, y, n, i, q, r, s)
    if x[i, MIN] == y[i, MAX] and x[i, MAX] > y[i, MIN]:
        if s > i + 1:
            i = s
        else:
            i += 1
            s = i
        return compute_domains_4(x, y, n, i, q, r, s)
    # u = 2
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_1(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int) -> int:
    while i < n and x[i, MIN] == y[i, MAX]:
        # enforce xi = yi
        x[i, MAX] = min(x[i, MAX], y[i, MAX])
        if x[i, MAX] < x[i, MIN]:
            return PROP_INCONSISTENCY
        y[i, MIN] = max(y[i, MIN], x[i, MIN])
        if y[i, MAX] < y[i, MIN]:
            return PROP_INCONSISTENCY
        i += 1
        q = i
    if i == n or x[i, MAX] < y[i, MIN]:
        return PROP_ENTAILMENT
    # enforce xq <= yq
    x[i, MAX] = min(x[i, MAX], y[i, MAX])
    if x[i, MAX] < x[i, MIN]:
        return PROP_INCONSISTENCY
    y[i, MIN] = max(y[i, MIN], x[i, MIN])
    if y[i, MAX] < y[i, MIN]:
        return PROP_INCONSISTENCY
    if r > i + 1:
        i = r
    else:
        i += 1
        r = i
    return compute_domains_2(x, y, n, i, q, r, s)


@njit(cache=True)
def compute_domains_lexicographic_leq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements lexicographic leq: x <_leq y.
    See https://www.diva-portal.org/smash/record.jsf?pid=diva2:1041533.
    :param domains: the domains of the variables,
           x is the list of the first n domains,
           y is the list of the last n domains
    :param parameters: unused here
    :return: the status of the propagation (consistency, inconsistency or entailement) as an int
    """
    # TODO: make incremental, use a var?
    n = len(domains) // 2
    return compute_domains_1(domains[:n], domains[n:], n, 0, 0, 0, 0)
