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
from numba import int64, njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_mod_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_mod_eq(n: int, variable: int, parameters: NDArray) -> int:
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
def compute_domains_mod_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x \\bmod y = z` with truncated division (the remainder takes the sign of the
    dividend x), i.e. the FlatZinc/MiniZinc ``int_mod`` semantics.

    :param domains: the domains of the variables, x is the first domain, y the second, z the third
    :type domains: NDArray
    :param parameters: unused here
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    z = domains[2]
    # y must be non-zero (division by zero is undefined): trim a zero bound
    if y[MIN] == 0:
        y[MIN] = 1
    if y[MAX] == 0:
        y[MAX] = -1
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    # both operands fixed: the remainder is determined
    if x[MIN] == x[MAX] and y[MIN] == y[MAX]:
        xi = int64(x[MIN])
        yi = int64(y[MIN])
        q = abs(xi) // abs(yi)
        if (xi < 0) != (yi < 0):
            q = -q
        r = xi - q * yi
        if r < z[MIN] or r > z[MAX]:
            return PROP_INCONSISTENCY
        z[MIN] = r
        z[MAX] = r
        return PROP_ENTAILMENT
    # |z| < |y| <= y_abs_max, and the remainder takes the sign of the dividend x
    y_abs_max = max(abs(int64(y[MIN])), abs(int64(y[MAX])))
    z_mag = y_abs_max - 1
    z_lo = 0 if x[MIN] >= 0 else -z_mag  # x cannot be negative -> z cannot be negative
    z_hi = 0 if x[MAX] <= 0 else z_mag  # x cannot be positive -> z cannot be positive
    if z_lo > z[MIN]:
        z[MIN] = z_lo
    if z_hi < z[MAX]:
        z[MAX] = z_hi
    if z[MIN] > z[MAX]:
        return PROP_INCONSISTENCY
    # conversely |y| > |z|: when z is bound away from 0 it lower-bounds |y|
    if z[MIN] > 0:
        z_abs_min = z[MIN]
    elif z[MAX] < 0:
        z_abs_min = -z[MAX]
    else:
        z_abs_min = 0
    if z_abs_min > 0:
        need = z_abs_min + 1  # |y| >= |z| + 1
        if y[MIN] > 0:  # y is positive
            if y[MIN] < need:
                y[MIN] = need
        elif y[MAX] < 0:  # y is negative
            if y[MAX] > -need:
                y[MAX] = -need
        if y[MIN] > y[MAX]:
            return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
