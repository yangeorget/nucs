import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_element_liv(n: int, parameters: NDArray) -> float:
    return n


def get_triggers_element_liv(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_element_liv(domains: NDArray, parameters: NDArray) -> int:
    """
    Enforces l_i = v.
    :param domains: the domains of the variables,
           l is the list of the first n-2 domains,
           i is the (n-1)th domain,
           v is the last domain
    :param parameters: the parameters of the propagator, it is unused
    """
    l = domains[:-2]
    i = domains[-2]
    i[MIN] = max(i[MIN], 0)
    i[MAX] = min(i[MAX], len(l) - 1)
    if i[MAX] < i[MIN]:
        return PROP_INCONSISTENCY
    v = domains[-1]
    v_min = sys.maxsize
    v_max = -sys.maxsize
    i_min = i[MIN]
    i_max = i[MAX]
    for idx in range(i[MIN], i[MAX] + 1):
        if v[MAX] < l[idx, MIN] or v[MIN] > l[idx, MAX]:  # no intersection
            if idx == i_min:
                i_min += 1
            elif idx == i_max:
                i_max -= 1
        else:  # intersection
            if l[idx, MIN] < v_min:
                v_min = l[idx, MIN]
            if l[idx, MAX] > v_max:
                v_max = l[idx, MAX]
    if i_max < i_min:
        return PROP_INCONSISTENCY
    i[MIN] = i_min
    i[MAX] = i_max
    v[MIN] = max(v[MIN], v_min)
    v[MAX] = min(v[MAX], v_max)
    if i_min == i_max:
        l[i_min, MIN] = max(l[i_min, MIN], v[MIN])
        l[i_max, MAX] = min(l[i_max, MAX], v[MAX])
        if v[MIN] == v[MAX]:
            return PROP_ENTAILMENT
    return PROP_CONSISTENCY
