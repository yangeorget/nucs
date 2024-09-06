import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.memory import MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY, new_triggers
from nucs.propagators.affine_eq_propagator import compute_domain_sum


def get_triggers_affine_leq(n: int, data: NDArray) -> NDArray:
    """
    Returns the triggers for this propagator.
    :param n: the number of variables
    :return: an array of triggers
    """
    triggers = new_triggers(n, False)
    for i, c in enumerate(data[:-1]):
        triggers[i, MIN] = c > 0
        triggers[i, MAX] = c < 0
    return triggers


@njit("int64(int32[::1,:], int32[:])", cache=True)
def compute_domains_affine_leq(domains: NDArray, a: NDArray) -> int:
    """
    Implements Sigma_i a_i * x_i <= a_{n-1}.
    :param domains: the domains of the variables
    :param a: the parameters of the propagator
    """
    domain_sum = compute_domain_sum(domains, a)
    if domain_sum[MIN] >= 0:
        return PROP_ENTAILMENT
    new_domains = np.copy(domains)
    for i, c in enumerate(a[:-1]):
        if c != 0:
            if c > 0:
                new_max = domains[i, MIN] + (domain_sum[MAX] // c)
                new_domains[i, MAX] = min(domains[i, MAX], new_max)
            else:
                new_min = domains[i, MAX] - (-domain_sum[MAX] // c)
                new_domains[i, MIN] = max(domains[i, MIN], new_min)
            if new_domains[i, MIN] > new_domains[i, MAX]:
                return PROP_INCONSISTENCY
    domains[:] = new_domains[:]  # numba does not support copyto
    return PROP_CONSISTENCY
