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


def get_complexity_diffn(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables (twice the number of rectangles)
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    return n * n


@njit(cache=True, fastmath=True)
def get_triggers_diffn(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever a coordinate bound changes.

    :param n: the number of variables
    :type n: int
    :param variable: the variable index, unused here
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray

    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True, fastmath=True)
def compute_domains_diffn(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements the 2D diffn (non-overlapping rectangles) constraint. Rectangle i has its bottom-left corner
    at ``(x_i, y_i)`` and constant size ``(dx_i, dy_i)``; no two rectangles overlap, i.e. for all i != j at
    least one of ``x_i + dx_i <= x_j``, ``x_j + dx_j <= x_i``, ``y_i + dy_i <= y_j``, ``y_j + dy_j <= y_i``
    holds.

    The first n variables are the x coordinates, the next n are the y coordinates. Filtering is pairwise: a
    pair that can no longer be separated along x (resp. y) must be separated along y (resp. x); when only one
    separation direction remains feasible it is enforced by tightening the coordinate bounds.

    :param domains: the domains of the variables, the n x coordinates then the n y coordinates
    :type domains: NDArray
    :param parameters: the n widths (dx) then the n heights (dy)
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    n = len(domains) // 2
    bound_nb = 0
    for v in range(2 * n):
        if domains[v, MIN] == domains[v, MAX]:
            bound_nb += 1
    for i in range(n):
        for j in range(i + 1, n):
            xi_min = domains[i, MIN]
            xi_max = domains[i, MAX]
            xj_min = domains[j, MIN]
            xj_max = domains[j, MAX]
            yi_min = domains[n + i, MIN]
            yi_max = domains[n + i, MAX]
            yj_min = domains[n + j, MIN]
            yj_max = domains[n + j, MAX]
            dxi = parameters[i]
            dxj = parameters[j]
            dyi = parameters[n + i]
            dyj = parameters[n + j]
            # A: i left of j, B: j left of i, C: i below j, D: j below i -- each is "still feasible?"
            a = xi_min + dxi <= xj_max
            b = xj_min + dxj <= xi_max
            c = yi_min + dyi <= yj_max
            d = yj_min + dyj <= yi_max
            x_sep = a or b
            y_sep = c or d
            if not x_sep and not y_sep:
                return PROP_INCONSISTENCY
            if not x_sep:
                # the rectangles overlap along x, so they must be separated along y
                if not c:  # only D remains: enforce y_j + dy_j <= y_i
                    if yj_min + dyj > domains[n + i, MIN]:
                        domains[n + i, MIN] = yj_min + dyj
                    if yi_max - dyj < domains[n + j, MAX]:
                        domains[n + j, MAX] = yi_max - dyj
                    if domains[n + i, MIN] > domains[n + i, MAX] or domains[n + j, MIN] > domains[n + j, MAX]:
                        return PROP_INCONSISTENCY
                elif not d:  # only C remains: enforce y_i + dy_i <= y_j
                    if yi_min + dyi > domains[n + j, MIN]:
                        domains[n + j, MIN] = yi_min + dyi
                    if yj_max - dyi < domains[n + i, MAX]:
                        domains[n + i, MAX] = yj_max - dyi
                    if domains[n + i, MIN] > domains[n + i, MAX] or domains[n + j, MIN] > domains[n + j, MAX]:
                        return PROP_INCONSISTENCY
            elif not y_sep:
                # the rectangles overlap along y, so they must be separated along x
                if not a:  # only B remains: enforce x_j + dx_j <= x_i
                    if xj_min + dxj > domains[i, MIN]:
                        domains[i, MIN] = xj_min + dxj
                    if xi_max - dxj < domains[j, MAX]:
                        domains[j, MAX] = xi_max - dxj
                    if domains[i, MIN] > domains[i, MAX] or domains[j, MIN] > domains[j, MAX]:
                        return PROP_INCONSISTENCY
                elif not b:  # only A remains: enforce x_i + dx_i <= x_j
                    if xi_min + dxi > domains[j, MIN]:
                        domains[j, MIN] = xi_min + dxi
                    if xj_max - dxi < domains[i, MAX]:
                        domains[i, MAX] = xj_max - dxi
                    if domains[i, MIN] > domains[i, MAX] or domains[j, MIN] > domains[j, MAX]:
                        return PROP_INCONSISTENCY
    # when every coordinate is fixed and no overlap was detected, the constraint can no longer be violated
    if bound_nb == 2 * n:
        return PROP_ENTAILMENT
    return PROP_CONSISTENCY
