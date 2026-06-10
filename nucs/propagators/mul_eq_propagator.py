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
from typing import Tuple

from numba import int64, njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_mul_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_mul_eq(n: int, variable: int, parameters: NDArray) -> int:
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
def prod_hull(xl: int, xu: int, yl: int, yu: int) -> Tuple[int, int]:
    """
    Returns the (min, max) hull of x * y over [xl, xu] x [yl, yu] (the bilinear extrema are at the corners).

    :return: the lower and upper product bounds
    :rtype: Tuple[int, int]
    """
    p1 = xl * yl
    p2 = xl * yu
    p3 = xu * yl
    p4 = xu * yu
    return min(min(p1, p2), min(p3, p4)), max(max(p1, p2), max(p3, p4))


@njit(cache=True, fastmath=True)
def div_lo(num_min: int, num_max: int, den_min: int, den_max: int) -> int:
    """
    Returns ceil(min(num / den)) over the corners of [num_min, num_max] x [den_min, den_max], for a
    denominator interval that does not contain 0. Since ceil is non-decreasing, ceil(min) = min(ceil).

    :return: the rounded-up lower quotient bound
    :rtype: int
    """
    return min(
        min(-((-num_min) // den_min), -((-num_min) // den_max)),
        min(-((-num_max) // den_min), -((-num_max) // den_max)),
    )


@njit(cache=True, fastmath=True)
def div_hi(num_min: int, num_max: int, den_min: int, den_max: int) -> int:
    """
    Returns floor(max(num / den)) over the corners of [num_min, num_max] x [den_min, den_max], for a
    denominator interval that does not contain 0. Since floor is non-decreasing, floor(max) = max(floor).

    :return: the rounded-down upper quotient bound
    :rtype: int
    """
    return max(
        max(num_min // den_min, num_min // den_max),
        max(num_max // den_min, num_max // den_max),
    )


@njit(cache=True, fastmath=True)
def compute_domains_mul_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x * y = z`.

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
    # int64 avoids int32 overflow of the corner products
    # z = x * y: tighten z to the hull of the four corner products
    z_lo, z_hi = prod_hull(int64(x[MIN]), int64(x[MAX]), int64(y[MIN]), int64(y[MAX]))
    if z_lo > z[MAX] or z_hi < z[MIN]:
        return PROP_INCONSISTENCY
    if z_lo > z[MIN]:
        z[MIN] = z_lo
    if z_hi < z[MAX]:
        z[MAX] = z_hi
    zl = int64(z[MIN])
    zu = int64(z[MAX])
    # x = z / y, only when 0 is not in [y] (otherwise x is unbounded)
    yl = int64(y[MIN])
    yu = int64(y[MAX])
    if yl > 0 or yu < 0:
        x_lo = div_lo(zl, zu, yl, yu)
        x_hi = div_hi(zl, zu, yl, yu)
        if x_lo > x[MAX] or x_hi < x[MIN]:
            return PROP_INCONSISTENCY
        if x_lo > x[MIN]:
            x[MIN] = x_lo
        if x_hi < x[MAX]:
            x[MAX] = x_hi
    # y = z / x, only when 0 is not in [x]
    xl = int64(x[MIN])
    xu = int64(x[MAX])
    if xl > 0 or xu < 0:
        y_lo = div_lo(zl, zu, xl, xu)
        y_hi = div_hi(zl, zu, xl, xu)
        if y_lo > y[MAX] or y_hi < y[MIN]:
            return PROP_INCONSISTENCY
        if y_lo > y[MIN]:
            y[MIN] = y_lo
        if y_hi < y[MAX]:
            y[MAX] = y_hi
    # re-tighten z now that x and y may have narrowed, so z is consistent before any entailment claim
    z_lo, z_hi = prod_hull(int64(x[MIN]), int64(x[MAX]), int64(y[MIN]), int64(y[MAX]))
    if z_lo > z[MAX] or z_hi < z[MIN]:
        return PROP_INCONSISTENCY
    if z_lo > z[MIN]:
        z[MIN] = z_lo
    if z_hi < z[MAX]:
        z[MAX] = z_hi
    x_ground = x[MIN] == x[MAX]
    y_ground = y[MIN] == y[MAX]
    # entailed once both factors are fixed (z is now their product), or once a factor is fixed to 0 (z is 0)
    if (x_ground and y_ground) or (x_ground and x[MIN] == 0) or (y_ground and y[MIN] == 0):
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
