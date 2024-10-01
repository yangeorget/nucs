import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_count_eq(n: int, parameters: NDArray) -> float:
    return 2 * n


def get_triggers_count_eq(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_count_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements Sigma_i (x_i == a) = x_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: the parameters of the propagator, a is the first parameter
    """
    x = domains[:-1]
    a = parameters[0]
    ok_count_max = len(x) - np.count_nonzero((x[:, MIN] > a) | (x[:, MAX] < a))
    ok_count_min = np.count_nonzero((x[:, MIN] == a) & (x[:, MAX] == a))
    # TODO: refactor (see exactly)
    counter = domains[-1]
    counter[MIN] = max(counter[MIN], ok_count_min)
    counter[MAX] = min(counter[MAX], ok_count_max)
    if counter[MIN] > counter[MAX]:
        return PROP_INCONSISTENCY
    if ok_count_min == ok_count_max:
        return PROP_ENTAILMENT
    if ok_count_min == counter[MAX]:  # we cannot have more domains equal to c
        x[(x[:, MIN] == a) & (x[:, MAX] > a), MIN] = a + 1
        x[(x[:, MIN] < a) & (x[:, MAX] == a), MAX] = a - 1
    if ok_count_max == counter[MIN]:  # we cannot have more domains different from c
        x[(x[:, MIN] <= a) & (a <= x[:, MAX]), :] = a
    return PROP_CONSISTENCY
