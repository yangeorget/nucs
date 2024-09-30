import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_exactly_eq(n: int, parameters: NDArray) -> float:
    return 2 * n


def get_triggers_exactly_eq(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_exactly_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i (x_i == a) = c.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is the first parameter, c is the second parameter
    """
    a = parameters[0]
    ok_count_max = len(domains) - np.count_nonzero((domains[:, MIN] > a) | (domains[:, MAX] < a))
    ok_count_min = np.count_nonzero((domains[:, MIN] == a) & (domains[:, MAX] == a))
    c = parameters[1]
    if ok_count_min > c or ok_count_max < c:
        return PROP_INCONSISTENCY
    if ok_count_min == c and ok_count_max == c:
        return PROP_ENTAILMENT
    if ok_count_min == c:  # we cannot have more domains equal to c
        domains[(domains[:, MIN] == a) & (domains[:, MAX] > a), MIN] = a + 1
        domains[(domains[:, MIN] < a) & (domains[:, MAX] == a), MAX] = a - 1
    if ok_count_max == c:  # we cannot have more domains different from c
        domains[(domains[:, MIN] <= a) & (a <= domains[:, MAX]), :] = a
    return PROP_CONSISTENCY
