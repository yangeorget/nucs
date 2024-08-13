import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, inconsistency, init_triggers
from ncs.propagators.affine_eq_propagator import compute_domain_sum


def get_triggers(n: int, data: NDArray) -> NDArray:
    triggers = init_triggers(n, False)
    for i in range(n):
        triggers[i, MIN] = data[i + 1] > 0
        triggers[i, MAX] = data[i + 1] < 0
    return triggers


@jit("int32[::1,:](int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> NDArray:
    """
    Implements Sigma_i a_{i+1} * x_i <= a_0.
    :param domains: the domains of the variables
    :return: the new domains
    """
    n = len(domains)
    domain_sum = compute_domain_sum(n, domains, data)
    new_domains = np.empty((2, len(domains)), dtype=np.int32).T
    new_domains[:] = domains[:]
    for i in range(n):
        ai = data[i + 1]
        if ai > 0:
            new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (domain_sum[MAX] // ai))
            if new_domains[i, MAX] < domains[i, MIN]:
                return inconsistency()
        elif ai < 0:
            new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (-domain_sum[MAX] // ai))
            if new_domains[i, MIN] > domains[i, MAX]:
                return inconsistency()
    return new_domains
