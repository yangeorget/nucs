from typing import Optional

import numpy as np
from numba import jit  # type: ignore
from numpy.typing import NDArray


def get_triggers(n: int, data: NDArray) -> NDArray:
    return np.ones((n, 2), dtype=bool)


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: Optional[NDArray] = None) -> Optional[NDArray]:
    """
    A propagator that does nothing.
    """
    return domains
