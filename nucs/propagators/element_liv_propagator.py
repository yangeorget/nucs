import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_element_liv(n: int, data: NDArray) -> float:
    return n


def get_triggers_element_liv(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_element_liv(domains: NDArray, data: NDArray) -> int:
    """
    Enforces l_i = x.
    :param domains: the domains of the variables
    :param data: the parameters of the propagator
    """
    l = domains[:-2]
    i = domains[-2]
    i[MIN] = max(i[MIN], 0)
    i[MAX] = min(i[MAX], len(l) - 1)
    if i[MAX] < i[MIN]:
        return PROP_INCONSISTENCY
    x = domains[-1]
    x_min = sys.maxsize
    x_max = -sys.maxsize
    i_min = i[MIN]
    i_max = i[MAX]
    for idx in range(i[MIN], i[MAX] + 1):
        if x[MAX] < l[idx, MIN] or x[MIN] > l[idx, MAX]:  # no intersection
            if idx == i_min:
                i_min += 1
            elif idx == i_max:
                i_max -= 1
        else:  # intersection
            if l[idx, MIN] < x_min:
                x_min = l[idx, MIN]
            if l[idx, MAX] > x_max:
                x_max = l[idx, MAX]
    if i_max < i_min:
        return PROP_INCONSISTENCY
    i[MIN] = i_min
    i[MAX] = i_max
    x[MIN] = max(x[MIN], x_min)
    x[MAX] = min(x[MAX], x_max)
    if i_min == i_max:
        l[i_min, MIN] = max(l[i_min, MIN], x[MIN])
        l[i_max, MAX] = min(l[i_max, MAX], x[MAX])
        if x[MIN] == x[MAX]:
            return PROP_ENTAILMENT
    return PROP_CONSISTENCY
