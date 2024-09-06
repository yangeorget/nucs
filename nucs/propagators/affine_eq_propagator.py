import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.memory import MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY, new_triggers


def get_triggers_affine_eq(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domain_sum(domains: NDArray, data: NDArray) -> NDArray:
    domain_sum = np.full(2, data[-1], dtype=np.int32)
    for i in range(len(data) - 1):
        c = data[i]
        if c > 0:
            domain_sum[MIN] -= domains[i, MAX] * c
            domain_sum[MAX] -= domains[i, MIN] * c
        elif c < 0:
            domain_sum[MIN] -= domains[i, MIN] * c
            domain_sum[MAX] -= domains[i, MAX] * c
    return domain_sum


@njit("int64(int32[::1,:], int32[:])", cache=True)
def compute_domains_affine_eq(domains: NDArray, a: NDArray) -> int:
    """
    Implements Sigma_i a_i * x_i = a_{n-1}.
    :param domains: the domains of the variables
    :param a: the parameters of the propagator
    """
    domain_sum = compute_domain_sum(domains, a)
    new_domains = np.copy(domains)
    for i, c in enumerate(a[:-1]):
        if c != 0:
            if c > 0:
                new_min = domains[i, MAX] - (domain_sum[MIN] // -c)
                new_max = domains[i, MIN] + (domain_sum[MAX] // c)
            else:
                new_min = domains[i, MAX] - (-domain_sum[MAX] // c)
                new_max = domains[i, MIN] + (-domain_sum[MIN] // -c)
            new_domains[i, MIN] = max(domains[i, MIN], new_min)
            new_domains[i, MAX] = min(domains[i, MAX], new_max)
            if new_domains[i, MIN] > new_domains[i, MAX]:
                return PROP_INCONSISTENCY
    domains[:] = new_domains[:]  # numba does not support copyto
    return PROP_CONSISTENCY
