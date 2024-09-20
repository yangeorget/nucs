from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_element_lic(n: int, data: NDArray) -> float:
    return n


def get_triggers_element_lic(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_element_lic(domains: NDArray, data: NDArray) -> int:
    """
    Enforces l_i = c.
    :param domains: the domains of the variables
    :param data: the parameters of the propagator
    """
    l = domains[:-1]
    i = domains[-1]
    # update i
    i[MIN] = max(i[MIN], 0)
    i[MAX] = min(i[MAX], len(l) - 1)
    if i[MAX] < i[MIN]:
        return PROP_INCONSISTENCY
    c = data[0]
    i_min = i[MIN]
    i_max = i[MAX]
    for idx in range(i[MIN], i[MAX] + 1):
        if c < l[idx, MIN] or c > l[idx, MAX]:  # no intersection
            if idx == i_min:
                i_min += 1
            elif idx == i_max:
                i_max -= 1
    if i_max < i_min:
        return PROP_INCONSISTENCY
    i[MIN] = i_min
    i[MAX] = i_max
    # when strict, update l
    # for idx in range(0, i[MIN]):
    #     if l[idx, MIN] == c:
    #         l[idx, MIN] = c + 1
    #     elif l[idx, MAX] == c:
    #         l[idx, MAX] = c - 1
    #     if l[idx, MIN] > l[idx, MAX]:
    #         return PROP_INCONSISTENCY
    # for idx in range(i[MAX] + 1, len(l)):
    #     if l[idx, MIN] == c:
    #         l[idx, MIN] = c + 1
    #     elif l[idx, MAX] == c:
    #         l[idx, MAX] = c - 1
    #     if l[idx, MIN] > l[idx, MAX]:
    #         return PROP_INCONSISTENCY
    if i_min == i_max:
        l[i_min, MIN] = l[i_max, MAX] = c
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
