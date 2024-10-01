import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_and(n: int, parameters: NDArray) -> float:
    return 3 * n


def get_triggers_and(n: int, parameters: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_and(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements & x_i = x_{n-1}.
    :param domains: the domains of the variables, x is an alias for domains
    :param parameters: unused here
    """
    x = domains[:-1]
    y = domains[-1]
    if np.any(x[:, MAX] == 0):
        y[MAX] = 0
    if np.all(x[:, MIN] == 1):
        y[MIN] = 1
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    if y[MIN] == 1:
        x[:, MIN] = 1
    if y[MAX] == 0:
        candidates_nb = 0
        candidate_idx = -1
        for i in range(len(x)):
            if x[i, MIN] == 0:
                candidate_idx = i
                candidates_nb += 1
        if candidates_nb == 1:
            x[candidate_idx, MAX] = 0
    return PROP_CONSISTENCY
