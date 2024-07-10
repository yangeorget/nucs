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
    c = data.item()
    new_domains = domains - (np.sum(domains, axis=0) - c)  # MIN and MAX are swapped
    if np.any(np.greater(new_domains[:, MAX], domains[:, MAX])) or np.any(
        np.less(new_domains[:, MIN], domains[:, MIN])
    ):
        return None
    tmp = np.maximum(new_domains[:, MAX], domains[:, MIN])
    new_domains[:, MAX] = np.minimum(new_domains[:, MIN], domains[:, MAX])
    new_domains[:, MIN] = tmp
    return new_domains
