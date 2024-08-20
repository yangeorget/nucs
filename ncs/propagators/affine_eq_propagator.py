import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY, new_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@jit(nopython=True, cache=True)
def compute_domain_sum(n: int, domains: NDArray, data: NDArray) -> NDArray:
    domain_sum = np.empty(2, dtype=np.int32)
    domain_sum[:] = data[-1]
    for i in range(n):
        c = data[i]
        if c > 0:
            domain_sum[MIN] -= domains[i, MAX] * c
            domain_sum[MAX] -= domains[i, MIN] * c
        elif c < 0:
            domain_sum[MIN] -= domains[i, MIN] * c
            domain_sum[MAX] -= domains[i, MAX] * c
    return domain_sum


@jit("int8(int32[::1,:], int32[:])", nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> np.int8:
    """
    Implements Sigma_i a_i * x_i = a_{n-1}.
    :param domains: the domains of the variables
    """
    n = len(domains)
    domain_sum = compute_domain_sum(n, domains, data)
    new_domains = np.empty_like(domains)
    new_domains[:] = domains[:]
    for i in range(n):
        if data[i] > 0:
            new_min = domains[i, MAX] - (domain_sum[MIN] // -data[i])
            new_max = domains[i, MIN] + (domain_sum[MAX] // data[i])
        elif data[i] < 0:
            new_min = domains[i, MAX] - (-domain_sum[MAX] // data[i])
            new_max = domains[i, MIN] + (-domain_sum[MIN] // -data[i])
        new_domains[i, MIN] = max(domains[i, MIN], new_min)
        new_domains[i, MAX] = min(domains[i, MAX], new_max)
        if new_domains[i, MIN] > new_domains[i, MAX]:
            return PROP_INCONSISTENCY
    domains[:] = new_domains[:]
    return PROP_CONSISTENCY
