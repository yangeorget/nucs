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

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_div_c_eq(n: int, parameters: NDArray) -> int:
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


@njit(cache=True)
def get_triggers_div_c_eq(n: int, variable: int, parameters: NDArray) -> int:
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


@njit(cache=True)
def _tdiv(x: int, m: int) -> int:
    """
    Returns the truncated (rounded toward zero) integer division of x by a positive divisor m.

    :param x: the dividend
    :type x: int
    :param m: the divisor, strictly positive
    :type m: int

    :return: x divided by m, rounded toward zero
    :rtype: int
    """
    if x >= 0:
        return x // m
    return -((-x) // m)


@njit(cache=True)
def compute_domains_div_c_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x \\div c = y` for a constant non-zero divisor c, with truncated division (the quotient
    is rounded toward zero), i.e. the FlatZinc/MiniZinc ``int_div`` semantics with a fixed divisor.

    The quotient is monotonic in x for a fixed c, so filtering is bound-consistent in a single pass: y is
    tightened to the image of x's bounds, then x is pruned back to the preimage of y's bounds. The magnitude
    of c is used throughout since ``x div c = -(x div |c|)`` for a negative c.

    :param domains: the domains of the variables, x is the first domain, y the second
    :type domains: NDArray
    :param parameters: the parameters, c is parameters[0]
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    y = domains[1]
    c = int(parameters[0])
    if c == 0:  # division by zero is undefined: the constraint holds for no value
        return PROP_INCONSISTENCY
    m = abs(c)  # x div c = -(x div m); work with the positive magnitude and flip the sign for c < 0
    # tighten y to the image of x's bounds; _tdiv is monotonic non-decreasing in x
    q_xmin = _tdiv(x[MIN], m)
    q_xmax = _tdiv(x[MAX], m)
    if c > 0:
        new_y_min = q_xmin
        new_y_max = q_xmax
    else:
        new_y_min = -q_xmax
        new_y_max = -q_xmin
    y[MIN] = max(y[MIN], new_y_min)
    y[MAX] = min(y[MAX], new_y_max)
    if y[MIN] > y[MAX]:
        return PROP_INCONSISTENCY
    # prune x to the preimage of y's bounds, mapping y back to the quotient q = x div m
    if c > 0:
        ql = y[MIN]
        qu = y[MAX]
    else:
        ql = -y[MAX]
        qu = -y[MIN]
    # smallest x with (x div m) >= ql is the lower edge of the ql block; largest x with (x div m) <= qu is
    # the upper edge of the qu block
    new_x_min = ql * m if ql > 0 else ql * m - m + 1
    new_x_max = qu * m + m - 1 if qu >= 0 else qu * m
    x[MIN] = max(x[MIN], new_x_min)
    x[MAX] = min(x[MAX], new_x_max)
    if x[MIN] > x[MAX]:
        return PROP_INCONSISTENCY
    return PROP_ENTAILMENT if x[MIN] == x[MAX] else PROP_CONSISTENCY
