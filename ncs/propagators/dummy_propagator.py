from numba import jit  # type: ignore
from numpy.typing import NDArray

from ncs.memory import init_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    return init_triggers(n, True)


@jit("int32[::1, :](int32[::1, :])", nopython=True, cache=True)
def compute_domains(domains: NDArray) -> NDArray:
    """
    A propagator that does nothing.
    """
    return domains
