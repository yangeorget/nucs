from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.utils import MAX, MIN


@jit(nopython=True, nogil=True)
def compute_domains(domains: NDArray) -> Optional[NDArray]:
    size = len(domains)
    new_domains = np.zeros((size, 2), dtype=np.int32)
    new_domains[0] = np.sum(domains[1:], axis=0)
    new_domains[1:, MIN] = domains[1:, MAX] + (domains[0, MIN] - new_domains[0, MAX])
    new_domains[1:, MAX] = domains[1:, MIN] + (domains[0, MAX] - new_domains[0, MIN])
    return new_domains
