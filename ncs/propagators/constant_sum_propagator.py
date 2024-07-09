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
    size = len(domains)
    new_domains = np.zeros((size, 2), dtype=np.int32)
    c = data.item()
    domain_sum = np.sum(domains, axis=0)
    new_domains[:, MIN] = domains[:, MAX] + (c - domain_sum[MAX])
    if np.any(np.greater(new_domains[:, MIN], domains[:, MAX])):
        return None
    new_domains[:, MAX] = domains[:, MIN] + (c - domain_sum[MIN])
    if np.any(np.less(new_domains[:, MAX], domains[:, MIN])):
        return None
    new_domains[:, MIN] = np.maximum(new_domains[:, MIN], domains[:, MIN])  # TODO: optimize ?
    new_domains[:, MAX] = np.minimum(new_domains[:, MAX], domains[:, MAX])
    return new_domains
