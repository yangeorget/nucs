import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_max_eq(n: int, data: NDArray) -> float:
    return 3 * n


def get_triggers_max_eq(n: int, data: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    triggers = new_triggers(n, True)
    triggers[-1, MIN] = False
    return triggers


@njit(cache=True)
def compute_domains_max_eq(domains: NDArray, data: NDArray) -> int:
    """
    Implements Max_i x_i = x_{n-1}.
    :param domains: the domains of the variables
    :param data: unused here
    """
    x = domains[:-1]
    y = domains[-1]
    y[MIN] = max(y[MIN], np.max(x[:, MIN]))
    y[MAX] = min(y[MAX], np.max(x[:, MAX]))
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    for i in range(len(x)):
        x[i, MAX] = min(x[i, MAX], y[MAX])
        if x[i, MAX] < x[i, MIN]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
