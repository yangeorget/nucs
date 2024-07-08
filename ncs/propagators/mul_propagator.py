from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.utils import MAX, MIN


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> Optional[NDArray]:
    """
    Enforces x_0 = x_1 * c_0
    :param domains: the domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    c = data.item()
    if c == 0:
        domains[0] = 0
        return domains
    new_domains = np.zeros((2, 2), dtype=np.int32)
    if c > 0:
        new_domains[0] = domains[1] * c
        new_domains[1, MIN] = -(domains[0, MIN] // -c)  # ceil division
        new_domains[1, MAX] = domains[0, MAX] // c  # floor division
    else:
        new_domains[0, MIN] = domains[1, MAX] * c
        new_domains[0, MAX] = domains[1, MIN] * c
        new_domains[1, MIN] = -(-domains[0, MAX] // c)
        new_domains[1, MAX] = -domains[0, MIN] // -c
    new_domains[:, MIN] = np.maximum(new_domains[:, MIN], domains[:, MIN])
    new_domains[:, MAX] = np.minimum(new_domains[:, MAX], domains[:, MAX])
    return new_domains
