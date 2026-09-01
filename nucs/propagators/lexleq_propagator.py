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
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)


def get_complexity_lexleq(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n


@njit(cache=True)
def get_triggers_lexleq(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an array of triggers
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_4(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int) -> int:
    while i < n and x[i, DOMAIN_MIN] == y[i, DOMAIN_MAX]:
        i += 1
        # s = i
    if i < n and x[i, DOMAIN_MIN] > y[i, DOMAIN_MAX]:
        # xq < yq
        x[q, DOMAIN_MAX] = min(x[q, DOMAIN_MAX], y[q, DOMAIN_MAX] - 1)
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        y[q, DOMAIN_MIN] = max(y[q, DOMAIN_MIN], x[q, DOMAIN_MIN] + 1)
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] < y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    # u = 4
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_3(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int) -> int:
    while i < n and x[i, DOMAIN_MAX] == y[i, DOMAIN_MIN]:
        i += 1
        # s = i
    if i == n or x[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        # xq <= yq
        x[q, DOMAIN_MAX] = min(x[q, DOMAIN_MAX], y[q, DOMAIN_MAX])
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        y[q, DOMAIN_MIN] = max(y[q, DOMAIN_MIN], x[q, DOMAIN_MIN])
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] <= y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    # u = 3
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_2(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int) -> int:
    while i < n and x[i, DOMAIN_MIN] == x[i, DOMAIN_MAX] == y[i, DOMAIN_MIN] == y[i, DOMAIN_MAX]:
        i += 1
        r = i
    if i == n or x[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        # xq <= yq
        x[q, DOMAIN_MAX] = min(x[q, DOMAIN_MAX], y[q, DOMAIN_MAX])
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        y[q, DOMAIN_MIN] = max(y[q, DOMAIN_MIN], x[q, DOMAIN_MIN])
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] <= y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    if x[i, DOMAIN_MIN] > y[i, DOMAIN_MAX]:
        # xq < yq
        x[q, DOMAIN_MAX] = min(x[q, DOMAIN_MAX], y[q, DOMAIN_MAX] - 1)
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        y[q, DOMAIN_MIN] = max(y[q, DOMAIN_MIN], x[q, DOMAIN_MIN] + 1)
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] < y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    if x[i, DOMAIN_MAX] == y[i, DOMAIN_MIN] and x[i, DOMAIN_MIN] < y[i, DOMAIN_MAX]:
        if s > i + 1:
            i = s
        else:
            i += 1
            s = i
        return compute_domains_3(x, y, n, i, q, r, s)
    if x[i, DOMAIN_MIN] == y[i, DOMAIN_MAX] and x[i, DOMAIN_MAX] > y[i, DOMAIN_MIN]:
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
    while i < n and x[i, DOMAIN_MIN] == y[i, DOMAIN_MAX]:
        # enforce xi = yi
        x[i, DOMAIN_MAX] = min(x[i, DOMAIN_MAX], y[i, DOMAIN_MAX])
        if x[i, DOMAIN_MAX] < x[i, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        y[i, DOMAIN_MIN] = max(y[i, DOMAIN_MIN], x[i, DOMAIN_MIN])
        if y[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        i += 1
        q = i
    if i == n or x[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        return PROP_ENTAILMENT
    # enforce xq <= yq
    x[i, DOMAIN_MAX] = min(x[i, DOMAIN_MAX], y[i, DOMAIN_MAX])
    if x[i, DOMAIN_MAX] < x[i, DOMAIN_MIN]:
        return PROP_INCONSISTENCY
    y[i, DOMAIN_MIN] = max(y[i, DOMAIN_MIN], x[i, DOMAIN_MIN])
    if y[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        return PROP_INCONSISTENCY
    if r > i + 1:
        i = r
    else:
        i += 1
        r = i
    return compute_domains_2(x, y, n, i, q, r, s)


@njit(cache=True)
def compute_domains_lexleq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements lexicographic leq: :math:`x <_leq y`.
    See https://www.diva-portal.org/smash/record.jsf?pid=diva2:1041533.

    :param domains: the domains of the variables,
           x is the list of the first n domains,
           y is the list of the last n domains
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    # TODO: make incremental, use a var?
    n = len(domains) >> 1
    return compute_domains_1(domains[:n], domains[n:], n, 0, 0, 0, 0)
