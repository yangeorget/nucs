from typing import Optional

from numba import jit  # type: ignore
from numpy.typing import NDArray


@jit(nopython=True, cache=True)
def compute_domains(domains: NDArray, data: Optional[NDArray] = None) -> Optional[NDArray]:
    """
    A propagator that does nothing.
    """
    return domains
