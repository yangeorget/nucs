import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import (
    MAX,
    MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    new_triggers,
)
from ncs.propagators.affine_eq_propagator import compute_domain_sum


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    triggers = new_triggers(n, False)
    for i in range(n):
        triggers[i, MIN] = data[i] > 0
        triggers[i, MAX] = data[i] < 0
    return triggers


@jit("int64(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> int:
    """
    Implements Sigma_i a_i * x_i <= a_{n-1}.
    :param domains: the domains of the variables
    """
    n = len(domains)
    domain_sum = compute_domain_sum(n, domains, data)
    if domain_sum[MIN] >= 0:
        return PROP_ENTAILMENT
    new_domains = np.empty_like(domains)
    new_domains[:] = domains[:]
    for i in range(n):
        if data[i] > 0:
            new_max = domains[i, MIN] + (domain_sum[MAX] // data[i])
            new_domains[i, MAX] = min(domains[i, MAX], new_max)
        elif data[i] < 0:
            new_min = domains[i, MAX] - (-domain_sum[MAX] // data[i])
            new_domains[i, MIN] = max(domains[i, MIN], new_min)
        if new_domains[i, MIN] > new_domains[i, MAX]:
            return PROP_INCONSISTENCY
    domains[:] = new_domains[:]
    return PROP_CONSISTENCY
