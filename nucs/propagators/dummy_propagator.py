from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import PROP_CONSISTENCY
from nucs.numpy import new_triggers


def get_complexity_dummy(n: int, parameters: NDArray) -> float:
    return 0


def get_triggers_dummy(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :return: an array of triggers
    """
    return new_triggers(n, True)


@njit(cache=True)
def compute_domains_dummy(domains: NDArray, parameters: NDArray) -> int:
    """
    A propagator that does nothing.
    """
    return PROP_CONSISTENCY
