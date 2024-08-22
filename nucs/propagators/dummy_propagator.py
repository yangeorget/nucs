from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.memory import PROP_CONSISTENCY, new_triggers


def get_triggers(n: int, data: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit("int64(int32[::1,:], int32[:])", cache=True)
def compute_domains(domains: NDArray, data: NDArray) -> int:
    """
    A propagator that does nothing.
    """
    return PROP_CONSISTENCY
