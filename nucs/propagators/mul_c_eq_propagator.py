###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024-2026 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY, PROP_ENTAILMENT


def get_complexity_mul_c_eq(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return 1


@njit(cache=True, fastmath=True)
def get_triggers_mul_c_eq(n: int, variable: int, parameters: NDArray) -> int:
    """
    Returns the triggers for this propagator.

    :param n: the number of variables
    :type n: int
    :param variable: the variable index
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_mul_c_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x * c = y`.

    :param domains: the domains of the variables, x is the first domain, y the second
    :type domains: NDArray
    :param parameters: the constant c, parameters[0]
    :type parameters: NDArray

    :return: the status of the propagation (consistency or inconsistency) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    c = int(parameters[0])
    if c == 0:
        if y[MIN] > 0 or y[MAX] < 0:
            return PROP_INCONSISTENCY
        y[:] = 0
        return PROP_ENTAILMENT
    # The fixpoint of x * c = y has a closed form: tighten x from y (with ceil/floor
    # rounding), then set y to the exact image c * x, which needs no further iteration.
    if c > 0:
        new_x_min = max(x[MIN], -((-y[MIN]) // c))
        new_x_max = min(x[MAX], y[MAX] // c)
    else:
        new_x_min = max(x[MIN], -((-y[MAX]) // c))
        new_x_max = min(x[MAX], y[MIN] // c)
    if new_x_min > new_x_max:
        return PROP_INCONSISTENCY
    x[MIN] = new_x_min
    x[MAX] = new_x_max
    if c > 0:
        y[MIN] = c * new_x_min
        y[MAX] = c * new_x_max
    else:
        y[MIN] = c * new_x_max
        y[MAX] = c * new_x_min
    return PROP_CONSISTENCY
