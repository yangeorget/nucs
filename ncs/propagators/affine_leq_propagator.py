from typing import Optional

from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.propagators.affine_eq_propagator import compute_domain_sum
from ncs.utils import MAX, MIN, init_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    triggers = init_triggers(n, False)
    for i in range(n):
        triggers[i, MIN] = data[i + 1] > 0
        triggers[i, MAX] = data[i + 1] < 0
    return triggers


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> Optional[NDArray]:
    """
    Implements Sigma_i a_{i+1} * x_i <= a_0.
    :param domains: the domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    n = len(domains)
    domain_sum = compute_domain_sum(n, domains, data)
    new_domains = domains.copy()
    for i in range(n):
        ai = data[i + 1]
        if ai > 0:
            new_domains[i, MAX] = min(new_domains[i, MAX], domains[i, MIN] + (domain_sum[MAX] // ai))
            if new_domains[i, MAX] < domains[i, MIN]:
                return None
        elif ai < 0:
            new_domains[i, MIN] = max(new_domains[i, MIN], domains[i, MAX] - (-domain_sum[MAX] // ai))
            if new_domains[i, MIN] > domains[i, MAX]:
                return None
    return new_domains
