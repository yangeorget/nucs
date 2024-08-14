import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, init_triggers
from ncs.propagators.affine_eq_propagator import compute_domain_sum


def get_triggers(n: int, data: NDArray) -> NDArray:
    triggers = init_triggers(n, False)
    for i in range(n):
        triggers[i, MIN] = data[i + 1] < 0
        triggers[i, MAX] = data[i + 1] > 0
    return triggers


@jit("boolean(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> bool:
    """
    Implements Sigma_i a_{i+1} * x_i >= a_0.
    :param domains: the domains of the variables
    :return: the new domains
    """
    n = len(domains)
    domain_sum = compute_domain_sum(n, domains, data)
    new_domains = np.empty_like(domains)
    new_domains[:] = domains[:]
    for i in range(n):
        ai = data[i + 1]
        if ai > 0:
            new_domains[i, MIN] = max(domains[i, MIN], domains[i, MAX] - (domain_sum[MIN] // -ai))
        elif ai < 0:
            new_domains[i, MAX] = min(domains[i, MAX], domains[i, MIN] + (-domain_sum[MIN] // -ai))
        if new_domains[i, MIN] > domains[i, MAX]:
            return False
    domains[:] = new_domains[:]
    return True
