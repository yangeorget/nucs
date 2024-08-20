import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY, new_triggers


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


@jit("int8(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> np.int8:
    """
    Implements Min_i x_i = x_{n-1}.
    :param domains: the domains of the variables
    """
    # TODO: implement entailment
    x = domains[:-1]
    y = domains[-1]
    y[MAX] = min(y[MAX], np.min(x[:, MAX]))
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    for i in range(len(x)):
        x[i, MIN] = max(x[i, MIN], y[MIN])
        if x[i, MAX] < x[i, MIN]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
