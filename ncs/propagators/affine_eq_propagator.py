from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import MAX, MIN, init_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    return init_triggers(n, True)


@jit(nopython=True, cache=True)
def compute_domain_sum(n: int, domains: NDArray, data: NDArray) -> NDArray:
    domain_sum = np.empty(2, dtype=np.int32)
    domain_sum[MIN] = data[0]
    domain_sum[MAX] = data[0]
    for i in range(n):
        ai = data[i + 1]
        if ai > 0:
            domain_sum[MIN] -= domains[i, MAX] * ai
            domain_sum[MAX] -= domains[i, MIN] * ai
        elif ai < 0:
            domain_sum -= domains[i] * ai
    return domain_sum


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> Optional[NDArray]:
    """
    Implements Sigma_i a_{i+1} * x_i >= a_0.
    :param domains: the domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    n = len(domains)
    domain_sum = compute_domain_sum(n, domains, data)
    new_domains = domains.copy()
    for i in range(n):
        ai = data[i + 1]
        if ai > 0:
            new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (domain_sum[MIN] // -ai))
            if new_domains[i, MIN] > domains[i, MAX]:
                return None
            new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (domain_sum[MAX] // ai))
            if new_domains[i, MAX] < domains[i, MIN]:
                return None
        elif ai < 0:
            new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (-domain_sum[MAX] // ai))
            if new_domains[i, MIN] > domains[i, MAX]:
                return None
            new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (-domain_sum[MIN] // -ai))
            if new_domains[i, MAX] < domains[i, MIN]:
                return None
    return new_domains
