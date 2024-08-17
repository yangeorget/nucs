import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, init_triggers
from ncs.propagators.affine_eq_propagator import compute_domain_sum


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    triggers = init_triggers(n, False)
    for i in range(n):
        triggers[i, MIN] = data[i] > 0
        triggers[i, MAX] = data[i] < 0
    return triggers


@jit("boolean(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> bool:
    """
    Implements Sigma_i a_i * x_i <= a_{n-1}.
    :param domains: the domains of the variables
    """
    n = len(domains)
    domain_sum = compute_domain_sum(n, domains, data)
    new_domains = np.empty_like(domains)
    new_domains[:] = domains[:]
    for i in range(n):
        if data[i] > 0:
            new_domains[i, MAX] = min(domains[i, MAX], domains[i, MIN] + (domain_sum[MAX] // data[i]))
        elif data[i] < 0:
            new_domains[i, MIN] = max(domains[i, MIN], domains[i, MAX] - (-domain_sum[MAX] // data[i]))
        if new_domains[i, MIN] > domains[i, MAX]:
            return False
    domains[:] = new_domains[:]
    return True
