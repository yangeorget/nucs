from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.utils import MAX, MIN


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> Optional[NDArray]:
    """
    :param domains: the domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    n = len(domains)
    # TODO: case where data[1:] is 0
    domain_sum = np.full(2, data[0], dtype=np.int32)
    for i in range(n):
        ai = data[i + 1]
        domain_sum[MIN] -= domains[i, MIN if ai < 0 else MAX] * ai
        domain_sum[MAX] -= domains[i, MAX if ai < 0 else MIN] * ai
    new_domains = domains.copy()
    for i in range(n):
        ai = data[i + 1]
        new_domains[i, MIN] = max(
            new_domains[i, MIN], domains[i, MAX] - (-domain_sum[MAX] // ai if ai < 0 else domain_sum[MIN] // -ai)
        )
        if new_domains[i, MIN] > domains[i, MAX]:
            return None
        new_domains[i, MAX] = min(
            new_domains[i, MAX], domains[i, MIN] + (-domain_sum[MIN] // -ai if ai < 0 else domain_sum[MAX] // ai)
        )
        if new_domains[i, MAX] < domains[i, MIN]:
            return None
    return new_domains
