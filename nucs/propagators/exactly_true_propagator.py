import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_exactly_true(n: int, parameters: NDArray) -> float:
    return 2 * n


def get_triggers_exactly_true(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_exactly_true(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i (x_i == 1) = c.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, c is the first parameter
    """
    ok_count_max = np.count_nonzero(domains[:, MAX] == 1)
    ok_count_min = np.count_nonzero((domains[:, MIN] == 1) & (domains[:, MAX] == 1))
    c = parameters[0]
    if ok_count_min > c or ok_count_max < c:
        return PROP_INCONSISTENCY
    if ok_count_min == c and ok_count_max == c:
        return PROP_ENTAILMENT
    if ok_count_min == c:  # we cannot have more domains equal to 1
        domains[(domains[:, MIN] == 0) & (domains[:, MAX] == 1), MAX] = 0
    elif ok_count_max == c:  # we cannot have more domains different from 1
        domains[(domains[:, MIN] == 0) & (domains[:, MAX] == 1), MIN] = 1
    return PROP_CONSISTENCY
