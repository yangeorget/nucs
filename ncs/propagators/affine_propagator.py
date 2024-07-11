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
        if ai > 0:
            domain_sum[MIN] -= domains[i, MAX] * ai
            domain_sum[MAX] -= domains[i, MIN] * ai
        elif ai < 0:
            domain_sum -= domains[i] * ai
    new_domains = np.empty((n, 2), dtype=np.int32)
    new_domains[:, MIN] = domains[:, MAX]
    new_domains[:, MAX] = domains[:, MIN]
    for i in range(n):
        ai = data[i + 1]
        if ai > 0:
            new_domains[i, MIN] += -(domain_sum[MIN] // -ai)  # ceil division
            new_domains[i, MAX] += domain_sum[MAX] // ai  # floor division
        elif ai < 0:
            new_domains[i, MIN] += -(-domain_sum[MAX] // ai)
            new_domains[i, MAX] += -domain_sum[MIN] // -ai
    if np.any(np.greater(new_domains[:, MIN], domains[:, MAX])) or np.any(
        np.less(new_domains[:, MAX], domains[:, MIN])
    ):
        return None
    new_domains[:, MIN] = np.maximum(new_domains[:, MIN], domains[:, MIN])
    new_domains[:, MAX] = np.minimum(new_domains[:, MAX], domains[:, MAX])
    return new_domains
