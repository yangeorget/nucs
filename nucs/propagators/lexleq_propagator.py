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
from collections.abc import Sequence

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


def get_state_lexleq(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the one cell it reports its changes in.

    :param n: the number of variables, unused here
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 1)
    :rtype: tuple[int, int]
    """
    return 0, 1


@njit(cache=True, inline="always")
def tighten_max(row: NDArray, value: int, prop_state: NDArray) -> None:
    """
    Narrows a domain's max, and records with the engine that something was written.

    The filtering here is spread over four mutually recursive functions, so the change is reported by
    *setting* the cell at each write rather than by threading a flag back through the returns:
    compute_domains_lexleq clears the cell on the way in -- overriding the 1 the engine pre-set -- and any
    write raises it again. That is the same answer, and it survives any control flow.

    :param row: the domain to narrow
    :type row: NDArray
    :param value: the candidate max
    :type value: int
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray
    """
    if value < row[DOMAIN_MAX]:
        row[DOMAIN_MAX] = value
        prop_state[0] = 1


@njit(cache=True, inline="always")
def tighten_min(row: NDArray, value: int, prop_state: NDArray) -> None:
    """
    Narrows a domain's min, and records with the engine that something was written.

    :param row: the domain to narrow
    :type row: NDArray
    :param value: the candidate min
    :type value: int
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray
    """
    if value > row[DOMAIN_MIN]:
        row[DOMAIN_MIN] = value
        prop_state[0] = 1


@njit(cache=True)
def compute_domains_4(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int, prop_state: NDArray) -> int:
    while i < n and x[i, DOMAIN_MIN] == y[i, DOMAIN_MAX]:
        i += 1
        # s = i
    if i < n and x[i, DOMAIN_MIN] > y[i, DOMAIN_MAX]:
        # xq < yq
        tighten_max(x[q], y[q, DOMAIN_MAX] - 1, prop_state)
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        tighten_min(y[q], x[q, DOMAIN_MIN] + 1, prop_state)
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] < y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    # u = 4
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_3(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int, prop_state: NDArray) -> int:
    while i < n and x[i, DOMAIN_MAX] == y[i, DOMAIN_MIN]:
        i += 1
        # s = i
    if i == n or x[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        # xq <= yq
        tighten_max(x[q], y[q, DOMAIN_MAX], prop_state)
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        tighten_min(y[q], x[q, DOMAIN_MIN], prop_state)
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] <= y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    # u = 3
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_2(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int, prop_state: NDArray) -> int:
    while i < n and x[i, DOMAIN_MIN] == x[i, DOMAIN_MAX] == y[i, DOMAIN_MIN] == y[i, DOMAIN_MAX]:
        i += 1
        r = i
    if i == n or x[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        # xq <= yq
        tighten_max(x[q], y[q, DOMAIN_MAX], prop_state)
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        tighten_min(y[q], x[q, DOMAIN_MIN], prop_state)
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] <= y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    if x[i, DOMAIN_MIN] > y[i, DOMAIN_MAX]:
        # xq < yq
        tighten_max(x[q], y[q, DOMAIN_MAX] - 1, prop_state)
        if x[q, DOMAIN_MAX] < x[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        tighten_min(y[q], x[q, DOMAIN_MIN] + 1, prop_state)
        if y[q, DOMAIN_MAX] < y[q, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        return PROP_ENTAILMENT if x[q, DOMAIN_MAX] < y[q, DOMAIN_MIN] else PROP_CONSISTENCY
    if x[i, DOMAIN_MAX] == y[i, DOMAIN_MIN] and x[i, DOMAIN_MIN] < y[i, DOMAIN_MAX]:
        if s > i + 1:
            i = s
        else:
            i += 1
            s = i
        return compute_domains_3(x, y, n, i, q, r, s, prop_state)
    if x[i, DOMAIN_MIN] == y[i, DOMAIN_MAX] and x[i, DOMAIN_MAX] > y[i, DOMAIN_MIN]:
        if s > i + 1:
            i = s
        else:
            i += 1
            s = i
        return compute_domains_4(x, y, n, i, q, r, s, prop_state)
    # u = 2
    return PROP_CONSISTENCY


@njit(cache=True)
def compute_domains_1(x: NDArray, y: NDArray, n: int, i: int, q: int, r: int, s: int, prop_state: NDArray) -> int:
    while i < n and x[i, DOMAIN_MIN] == y[i, DOMAIN_MAX]:
        # enforce xi = yi
        tighten_max(x[i], y[i, DOMAIN_MAX], prop_state)
        if x[i, DOMAIN_MAX] < x[i, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        tighten_min(y[i], x[i, DOMAIN_MIN], prop_state)
        if y[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
            return PROP_INCONSISTENCY
        i += 1
        q = i
    if i == n or x[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        return PROP_ENTAILMENT
    # enforce xq <= yq
    tighten_max(x[i], y[i, DOMAIN_MAX], prop_state)
    if x[i, DOMAIN_MAX] < x[i, DOMAIN_MIN]:
        return PROP_INCONSISTENCY
    tighten_min(y[i], x[i, DOMAIN_MIN], prop_state)
    if y[i, DOMAIN_MAX] < y[i, DOMAIN_MIN]:
        return PROP_INCONSISTENCY
    if r > i + 1:
        i = r
    else:
        i += 1
        r = i
    return compute_domains_2(x, y, n, i, q, r, s, prop_state)


@njit(cache=True)
def compute_domains_lexleq(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements lexicographic leq: :math:`x <_leq y`.
    See https://www.diva-portal.org/smash/record.jsf?pid=diva2:1041533.

    :param domains: the domains of the variables,
           x is the list of the first n domains,
           y is the list of the last n domains
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray
    :param prop_state: this propagator's state block, whose first cell is the change report
    :type prop_state: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    # TODO: make incremental, use a var?
    n = len(domains) >> 1
    prop_state[0] = 0  # cleared here and raised again by any write, see tighten_max
    return compute_domains_1(domains[:n], domains[n:], n, 0, 0, 0, 0, prop_state)
