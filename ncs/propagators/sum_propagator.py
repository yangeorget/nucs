from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.utils import MAX, MIN


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: Optional[NDArray] = None) -> Optional[NDArray]:
    """
    Enforces x_0 = Sigma_i>0 x_i
    :param domains: the domains of the variables
    :return: the new domains or None if an inconsistency is detected
    """
    size = len(domains)
    new_domains = np.zeros((size, 2), dtype=np.int32)
    new_domains[0] = np.sum(domains[1:], axis=0)
    new_domains[1:, MIN] = domains[1:, MAX] + (domains[0, MIN] - new_domains[0, MAX])
    if np.any(np.greater(new_domains[:, MIN], domains[:, MAX])):
        return None
    new_domains[1:, MAX] = domains[1:, MIN] + (domains[0, MAX] - new_domains[0, MIN])
    if np.any(np.less(new_domains[:, MAX], domains[:, MIN])):
        return None
    new_domains[:, MIN] = np.maximum(new_domains[:, MIN], domains[:, MIN])  # TODO: optimize ?
    new_domains[:, MAX] = np.minimum(new_domains[:, MAX], domains[:, MAX])
    return new_domains
