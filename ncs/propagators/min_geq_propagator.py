import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import (
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    new_triggers,
)


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    triggers = new_triggers(n, False)
    for i in range(n - 1):
        triggers[i, MAX] = True
    triggers[-1, MIN] = True
    return triggers


@njit("int64(int32[::1,:], int32[:])", cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> int:
    """
    Implements Min_i x_i >= x_{n-1}.
    :param domains: the domains of the variables
    :param data: unused here
    """
    x = domains[:-1]
    y = domains[-1]
    if y[MAX] <= np.min(x[:, MIN]):
        return PROP_ENTAILMENT
    y[MAX] = min(y[MAX], np.min(x[:, MAX]))
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    for i in range(len(x)):
        x[i, MIN] = max(x[i, MIN], y[MIN])
        if x[i, MAX] < x[i, MIN]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
