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

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_ENTAILMENT, PROP_INCONSISTENCY


def get_complexity_mod_c_eq(n: int, parameters: NDArray) -> int:
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
def get_triggers_mod_c_eq(n: int, variable: int, parameters: NDArray) -> int:
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
def _res_minmax(a: int, b: int, lo: int, hi: int, m: int) -> Tuple[bool, int, int]:
    """
    Returns ``(found, rmin, rmax)``: the smallest and largest remainder achievable as v % m for some v in the
    non-negative range [a, b], restricted to the window [lo, hi] (0 <= lo <= hi < m).

    The achievable remainders of [a, b] form the full period when the range spans at least m integers, a
    single interval [a % m, b % m] when it stays within one block, and otherwise two intervals
    ([0, b % m] and [a % m, m - 1]) -- the wrap leaves a hole, so the hull would be too loose.
    """
    if b - a >= m - 1:  # spans a full period: every remainder is reachable
        return True, lo, hi
    ra = a % m
    rb = b % m
    if ra <= rb:  # single interval [ra, rb]
        n_lo = max(ra, lo)
        n_hi = min(rb, hi)
        if n_lo <= n_hi:
            return True, n_lo, n_hi
        return False, 0, 0
    # wraps: [0, rb] and [ra, m - 1]
    found = False
    rmin = 0
    rmax = 0
    if lo <= rb:  # intersect [0, rb] with [lo, hi]
        rmin = lo
        rmax = min(rb, hi)
        found = True
    if hi >= ra:  # intersect [ra, m - 1] with [lo, hi]
        hi2 = hi  # min(m - 1, hi) == hi since hi < m
        lo2 = max(ra, lo)
        if found:
            rmax = hi2
        else:
            rmin = lo2
            rmax = hi2
            found = True
    return found, rmin, rmax


@njit(cache=True, fastmath=True)
def _first_ge(start: int, rl: int, ru: int, m: int) -> int:
    """
    Returns the smallest v >= start (start >= 0) whose remainder v % m lies in [rl, ru] (0 <= rl <= ru < m).
    """
    r = start % m
    if r < rl:
        return start + (rl - r)
    if r <= ru:
        return start
    return start + (m - r) + rl  # the next block's rl


@njit(cache=True, fastmath=True)
def _last_le(end: int, rl: int, ru: int, m: int) -> int:
    """
    Returns the largest v <= end (end >= 0) whose remainder v % m lies in [rl, ru] (0 <= rl <= ru < m).
    """
    r = end % m
    if r > ru:
        return end - (r - ru)
    if r >= rl:
        return end
    return end - r - (m - ru)  # the previous block's ru


@njit(cache=True, fastmath=True)
def compute_domains_mod_c_eq(domains: NDArray, parameters: NDArray) -> int:
    """
    Implements :math:`x \\bmod m = z` for a constant modulus m, with truncated division (the remainder takes
    the sign of the dividend x), i.e. the FlatZinc/MiniZinc ``int_mod`` semantics with a fixed divisor.

    The remainder is independent of the sign of m, so the modulus magnitude is used throughout and the
    negative part of x is mirrored onto the non-negative remainder arithmetic. Filtering is bound-consistent.

    :param domains: the domains of the variables, x is the first domain, z the second
    :type domains: NDArray
    :param parameters: the parameters, m is parameters[0]
    :type parameters: NDArray

    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    :rtype: int
    """
    x = domains[0]
    z = domains[1]
    m = abs(parameters[0])  # the remainder is the same for m and -m
    if m == 1:  # x mod 1 is always 0
        if z[MIN] > 0 or z[MAX] < 0:
            return PROP_INCONSISTENCY
        z[MIN] = 0
        z[MAX] = 0
        return PROP_ENTAILMENT if x[MIN] == x[MAX] else PROP_CONSISTENCY
    xl = x[MIN]
    xu = x[MAX]
    # tighten z to the achievable remainders of (x mod m) over [xl, xu] that fall inside z, handling the
    # non-negative and negative parts of x separately (the remainder mirrors the sign of x)
    found = False
    z_lo = 0
    z_hi = 0
    if xu >= 0:  # non-negative part [max(xl, 0), xu], remainder in [0, m - 1]
        lo = max(z[MIN], 0)
        hi = min(z[MAX], m - 1)
        if lo <= hi:
            ok, rmin, rmax = _res_minmax(xl if xl > 0 else 0, xu, lo, hi, m)
            if ok:
                z_lo = rmin
                z_hi = rmax
                found = True
    if xl < 0:  # negative part: z = -(x' mod m) for x' = -x in [-min(xu, -1), -xl]
        lo = max(-z[MAX], 0)
        hi = min(-z[MIN], m - 1)
        if lo <= hi:
            ok, rmin, rmax = _res_minmax(-(xu if xu < -1 else -1), -xl, lo, hi, m)
            if ok:
                n_lo = -rmax
                n_hi = -rmin
                if found:
                    z_lo = min(z_lo, n_lo)
                    z_hi = max(z_hi, n_hi)
                else:
                    z_lo = n_lo
                    z_hi = n_hi
                    found = True
    if not found:
        return PROP_INCONSISTENCY
    z[MIN] = z_lo
    z[MAX] = z_hi
    # prune x to the values whose remainder lands in the tightened z, again by part
    x_min = 0
    x_max = 0
    feasible = False
    if xl < 0:  # negative part
        sl = max(-z[MAX], 0)
        su = min(-z[MIN], m - 1)
        if sl <= su:
            ap = -(xu if xu < -1 else -1)
            bp = -xl
            lo_xp = _first_ge(ap, sl, su, m)
            if lo_xp <= bp:
                hi_xp = _last_le(bp, sl, su, m)
                x_min = -hi_xp
                x_max = -lo_xp
                feasible = True
    if xu >= 0:  # non-negative part
        rl = max(z[MIN], 0)
        ru = min(z[MAX], m - 1)
        if rl <= ru:
            a0 = xl if xl > 0 else 0
            lo_x = _first_ge(a0, rl, ru, m)
            if lo_x <= xu:
                hi_x = _last_le(xu, rl, ru, m)
                if feasible:
                    x_min = min(x_min, lo_x)
                    x_max = max(x_max, hi_x)
                else:
                    x_min = lo_x
                    x_max = hi_x
                    feasible = True
    if not feasible:
        return PROP_INCONSISTENCY
    if x_min > x[MIN]:
        x[MIN] = x_min
    if x_max < x[MAX]:
        x[MAX] = x_max
    return PROP_ENTAILMENT if x[MIN] == x[MAX] else PROP_CONSISTENCY
