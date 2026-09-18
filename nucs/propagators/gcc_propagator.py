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
import math
from collections.abc import Sequence

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_MIN_MAX,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
)
from nucs.propagators.alldifferent_propagator import argsort_into, argsort_into_warm, path_max, path_min, path_set


def get_complexity_gcc(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables
    :type n: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray
    :return: an int
    :rtype: int
    """
    return int(n * math.log(n))


def get_state_gcc(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the cell it reports its changes in, a persistent
    cold/warm flag, the two partial-sum
    tables built from the capacities, and the scratch space compute_domains_gcc used to allocate with
    np.empty/np.zeros on every call.

    The two partial-sum tables are a function of parameters alone, which the engine never writes, so they
    are built once on the cold call instead of on every one. Every other cell is either fully overwritten
    before it is read (bounds, t, d, h, ranks), explicitly re-zeroed by compute_domains_gcc itself
    (stable_intervals, stable_sets, new_mins, which used to come from a fresh np.zeros) or a stale
    permutation, which is still a permutation (the sort permutations, warm-started like alldifferent's).
    The entailment test's value counts come last. So the whole block is an untrailed hint: staleness costs time,
    never correctness.

    :param n: the number of variables
    :type n: int
    :param parameters: the domain offset, then the lower capacities, then the upper capacities
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb) = (0, 2 + 4 * (m + 6) + 6 * bounds_nb + 5n + 2m + 1)
    :rtype: tuple[int, int]
    """
    bounds_nb = 2 * (n + 1)
    m = (len(parameters) - 1) >> 1  # number of values
    psum_nb = 2 * (m + 6)  # cells of one (2, m + 6) partial_sum table, as laid out by init_partial_sum_into
    return 0, 2 + 2 * psum_nb + 6 * bounds_nb + 5 * n + 2 * m + 1


@njit(cache=True)
def get_triggers_gcc(n: int, variable: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.

    :param n: the number of variables
    :type n: int
    :param variable: the index of the variable
    :type variable: int
    :param parameters: the parameters, unused here
    :type parameters: NDArray
    :return: an event mask
    :rtype: int
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def init_partial_sum_into(partial_sum: NDArray, first_value: int, m: int, values: NDArray) -> None:
    """
    Inits the partial_sum data structure:
    ---------------------
    | sm | first_value |
    ---------------------
    | ds  | last_value  |
    ---------------------

    Writes into a caller-provided (2, m + 6) table rather than allocating one, so that a propagator whose
    capacities never change builds it once instead of on every call. The ds row is walked rather than
    filled, so the table has to arrive zeroed -- which is what the np.zeros this replaced gave it for free.

    :param partial_sum: the zeroed (2, m + 6) table to fill
    :type partial_sum: NDArray
    :param first_value: the first domain value
    :type first_value: int
    :param m: the number of values
    :type m: int
    :param values: the capacities
    :type values: NDArray
    """
    partial_sum[0, -1] = first_value - 3
    partial_sum[1, -1] = first_value + m + 1
    sm = partial_sum[0, :-1]
    sm[0] = 0
    sm[1] = 1
    sm[2] = 2
    for i in range(2, m + 2):
        sm[i + 1] = sm[i] + values[i - 2]
    sm[m + 3] = sm[m + 2] + 1
    sm[m + 4] = sm[m + 3] + 1
    ds = partial_sum[1, :-1]
    i = m + 3
    j = m + 4
    while i > 0:
        while sm[i] == sm[i - 1]:
            ds[i] = j
            i -= 1
        ds[j] = i
        j = i
        i -= 1
    ds[j] = 0


@njit(cache=True)
def get_sum(psum: NDArray, start: int, end: int) -> int:
    fv = psum[0, -1]
    sum = psum[0, :-1]
    if start <= end:
        # assert fv <= start
        # assert end <= get_last_value(psum)
        return sum[end - fv] - sum[start - fv - 1]
    else:
        # assert fv <= end
        # assert start <= get_last_value(psum)
        return sum[end - fv - 1] - sum[start - fv]


@njit(cache=True)
def get_min_value(psum: NDArray) -> int:
    return psum[0, -1] + 3


@njit(cache=True)
def get_max_value(psum: NDArray) -> int:
    return psum[1, -1] - 2


@njit(cache=True)
def skip_non_null_elements_right(psum: NDArray, value: int) -> int:
    value -= psum[0, -1]
    ps = psum[1, value]
    return (max(ps, value)) + psum[0, -1]


@njit(cache=True)
def skip_non_null_elements_left(psum: NDArray, value: int) -> int:
    value -= psum[0, -1]
    ps = psum[1, value]
    return (psum[1, ps] if ps > value else value) + psum[0, -1]


@njit(cache=True)
def update_bounds(
    bounds: NDArray,
    n: int,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    max_sorted_vars: NDArray,
    l: NDArray,
    u: NDArray,
) -> int:
    min_value = domains[min_sorted_vars[0], DOMAIN_MIN]
    max_value = domains[max_sorted_vars[0], DOMAIN_MAX] + 1
    bounds[0] = last = l[0, -1] + 1
    i = j = nb = 0
    while True:
        if i < n and min_value <= max_value:
            if min_value != last:
                nb += 1
                bounds[nb] = last = min_value
            ranks[min_sorted_vars[i], DOMAIN_MIN] = nb
            i += 1
            if i < n:
                min_value = domains[min_sorted_vars[i], DOMAIN_MIN]
        else:
            if max_value != last:
                nb += 1
                bounds[nb] = last = max_value
            ranks[max_sorted_vars[j], DOMAIN_MAX] = nb
            j += 1
            if j == n:
                break
            max_value = domains[max_sorted_vars[j], DOMAIN_MAX] + 1
    bounds[nb + 1] = u[1, -1] + 1
    return nb


@njit(cache=True)
def filter_lower_max(
    n: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    max_sorted_vars: NDArray,
    u: NDArray,
    prop_state: NDArray,
) -> bool:
    for i in range(1, nb + 2):
        i1 = i - 1
        t[i] = h[i] = i1
        d[i] = get_sum(u, bounds[i1], bounds[i] - 1)
        if d[i] == 0:
            # an interval of values with no capacity is full from the start: linked as the loop below links
            # one it fills, it is skipped rather than driven below zero, which breaks the Hall test
            t[i] = i + 1
    for i in range(len(max_sorted_vars)):
        x = ranks[max_sorted_vars[i], DOMAIN_MIN]
        y = ranks[max_sorted_vars[i], DOMAIN_MAX]
        z = path_max(t, x + 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z + 1
            z = path_max(t, t[z])
            t[z] = j
        delta = d[z] - get_sum(u, bounds[y], bounds[z] - 1)
        if delta < 0:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x + 1, z, z)  # path compression
        if h[x] > x:
            w = path_max(h, h[x])
            variable = max_sorted_vars[i]
            if domains[variable, DOMAIN_MIN] != bounds[w]:
                domains[variable, DOMAIN_MIN] = bounds[w]
                prop_state[0] = 1
            path_set(h, x, w, w)  # path compression
            # changes = 1
        if delta == 0:
            j1 = j - 1
            path_set(h, h[y], j1, y)  # mark hall interval
            h[y] = j1  # hall interval[bounds[j], bounds[y]]
    return True


@njit(cache=True)
def filter_upper_max(
    n: int,
    nb: int,
    t: NDArray,
    d: NDArray,
    h: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    u: NDArray,
    prop_state: NDArray,
) -> bool:
    for i in range(nb + 1):
        i1 = i + 1
        t[i] = h[i] = i1
        d[i] = get_sum(u, bounds[i], bounds[i1] - 1)
        if d[i] == 0:
            t[i] = i - 1  # full from the start, as in filter_lower_max
    for i in range(n - 1, -1, -1):
        x = ranks[min_sorted_vars[i], DOMAIN_MAX]
        y = ranks[min_sorted_vars[i], DOMAIN_MIN]
        z = path_min(t, x - 1)
        j = t[z]
        d[z] -= 1
        if d[z] == 0:
            t[z] = z - 1
            z = path_min(t, t[z])
            t[z] = j
        delta = d[z] - get_sum(u, bounds[z], bounds[y] - 1)
        if delta < 0:  # moved above the path compression which is not the case in the paper
            return False
        path_set(t, x - 1, z, z)  # path compression
        if h[x] < x:
            w = path_min(h, h[x])
            variable = min_sorted_vars[i]
            if domains[variable, DOMAIN_MAX] != bounds[w] - 1:
                domains[variable, DOMAIN_MAX] = bounds[w] - 1
                prop_state[0] = 1
            path_set(h, x, w, w)  # path compression
            # changes = 1
        if delta == 0:
            j1 = j + 1
            path_set(h, h[y], j1, y)  # mark hall interval
            h[y] = j1  # hall interval[bounds[j], bounds[y]]
    return True


@njit(cache=True)
def filter_lower_min(
    n: int,
    nb: int,
    tl: NDArray,
    c: NDArray,
    sets: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    max_sorted_vars: NDArray,
    l: NDArray,
    stable_intervals: NDArray,
    stable_sets: NDArray,
    new_mins: NDArray,
    prop_state: NDArray,
) -> bool:
    w = nb + 1
    for i in range(nb + 1, 0, -1):
        stable_sets[i] = stable_intervals[i] = i - 1
        c[i] = get_sum(l, bounds[i - 1], bounds[i] - 1)
        if c[i] == 0:  # if the capacity between both bounds is zero, we have an unstable set between these two bounds
            sets[i - 1] = w
        else:
            sets[w] = i - 1
            w = i - 1
    w = nb + 1
    for i in range(nb + 1, -1, -1):
        if c[i] == 0:
            tl[i] = w
        else:
            tl[w] = w = i
    for i in range(len(max_sorted_vars)):  # visit intervals in increasing max order
        x = ranks[max_sorted_vars[i], DOMAIN_MIN]
        y = ranks[max_sorted_vars[i], DOMAIN_MAX]
        z = path_max(tl, x + 1)
        j = tl[z]
        if z != x + 1:
            # If bounds[z] - 1 belongs to a stable set, [bounds[x], bounds[z]) is a sub set of this stable set.
            w = path_max(stable_sets, x + 1)
            v = stable_sets[w]
            path_set(stable_sets, x + 1, w, w)  # path compression
            w = min(y, z)
            path_set(stable_sets, stable_sets[w], v, w)
            stable_sets[w] = v
        if c[z] <= get_sum(l, bounds[y], bounds[z] - 1):
            # (potentialStableSets[y], y] is a stable set
            w = path_max(stable_intervals, stable_sets[y])
            path_set(stable_intervals, stable_sets[y], w, w)  # path compression
            v = stable_intervals[w]
            path_set(stable_intervals, stable_intervals[y], v, y)
            stable_intervals[y] = v
        else:
            c[z] -= 1  # decrease the capacity between the two bounds
            if c[z] == 0:
                tl[z] = z + 1
                z = path_max(tl, tl[z])
                tl[z] = j
            # If the lower bound belongs to an unstable or a stable set, remind the new value we might assign to
            # the lower bound in case the variable doesn't belong to a stable set.
            if sets[x] > x:
                w = path_max(sets, x)
                new_mins[i] = w
                path_set(sets, x, w, w)  # path compression
            else:
                new_mins[i] = x  # do not shrink the variable
            if c[z] == get_sum(l, bounds[y], bounds[z] - 1):  # if an unstable set is discovered
                # consider stable and unstable sets beyond y (pathmax; the path is fully compressed)
                y = max(y, sets[y])
                path_set(sets, sets[y], j - 1, y)  # mark the new unstable set
                sets[y] = j - 1
        path_set(tl, x + 1, z, z)  # path compression
    if sets[nb] != 0:  # if there is a failure set
        return False
    # Perform path compression over all elements in the stable interval data structure. This data structure will no
    # longer be modified and will be accessed n or 2n times. Therefore, we can afford a linear time compression.
    for i in range(nb + 1, 0, -1):
        if stable_intervals[i] > i:
            stable_intervals[i] = w
        else:
            w = i
    # For all variables that are not a subset of a stable set, shrink the lower bound.
    for i in range(n - 1, -1, -1):
        x = ranks[max_sorted_vars[i], DOMAIN_MIN]
        y = ranks[max_sorted_vars[i], DOMAIN_MAX]
        if stable_intervals[x] <= x or y > stable_intervals[x]:
            variable = max_sorted_vars[i]
            new_min = skip_non_null_elements_right(l, bounds[new_mins[i]])
            if domains[variable, DOMAIN_MIN] != new_min:
                domains[variable, DOMAIN_MIN] = new_min
                prop_state[0] = 1
            # changes = 1
    return True


@njit(cache=True)
def filter_upper_min(
    n: int,
    nb: int,
    tl: NDArray,
    c: NDArray,
    sets: NDArray,
    bounds: NDArray,
    domains: NDArray,
    ranks: NDArray,
    min_sorted_vars: NDArray,
    l: NDArray,
    stable_intervals: NDArray,
    new_maxs: NDArray,
    prop_state: NDArray,
) -> bool:
    w = 0
    for i in range(nb + 1):
        c[i] = get_sum(l, bounds[i], bounds[i + 1] - 1)
        if c[i] == 0:  # if the capacity between both bounds is zero, we have an unstable set between these two bounds
            tl[i] = w
        else:
            tl[w] = w = i
    tl[w] = nb + 1
    w = 0
    for i in range(1, nb + 1):
        if c[i - 1] == 0:
            sets[i] = w
        else:
            sets[w] = i
            w = i
    sets[w] = nb + 1
    for i in range(n - 1, -1, -1):  # visit intervals in decreasing max order
        x = ranks[min_sorted_vars[i], DOMAIN_MAX]
        y = ranks[min_sorted_vars[i], DOMAIN_MIN]
        # solve the lower bound problem
        z = path_min(tl, x - 1)
        j = tl[z]
        # If the variable is not in a discovered stable set
        # Possible optimization: use the array stbl_intervals to perform this test
        if c[z] > get_sum(l, bounds[z], bounds[y] - 1):
            c[z] -= 1
            if c[z] == 0:
                tl[z] = z - 1
                z = path_min(tl, tl[z])
                tl[z] = j
            if sets[x] < x:
                w = path_min(sets, sets[x])
                new_maxs[i] = w
                path_set(sets, x, w, w)  # path compression
            else:
                new_maxs[i] = x
            if c[z] == get_sum(l, bounds[z], bounds[y] - 1):
                y = min(y, sets[y])
                path_set(sets, sets[y], j + 1, y)  # loop
                sets[y] = j + 1
        path_set(tl, x - 1, z, z)
    #  For all variables that are not subsets of a stable set, shrink the lower bound.
    for i in range(n - 1, -1, -1):
        x = ranks[min_sorted_vars[i], DOMAIN_MIN]
        y = ranks[min_sorted_vars[i], DOMAIN_MAX]
        if stable_intervals[x] <= x or y > stable_intervals[x]:
            variable = min_sorted_vars[i]
            new_max = skip_non_null_elements_left(l, bounds[new_maxs[i]] - 1)
            if domains[variable, DOMAIN_MAX] != new_max:
                domains[variable, DOMAIN_MAX] = new_max
                prop_state[0] = 1
            # changes = 1
    return True


def is_vacuous_gcc(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool:
    """
    Returns whether the parameters make the constraint vacuous, whatever the domains.

    A value that no variable is required to take (lower capacity 0) and that all of them may take (upper
    capacity at least n) constrains nothing. When that holds for every value, every assignment satisfies the
    constraint, so it need never be posted. The upper capacities are scanned first because that is what fails
    fast on a constraint that does bind. MiniZinc produces these in quantity: global_cardinality_low_up leaves
    the values outside its cover unconstrained, and a cover whose own capacities do not bite makes the lot
    vacuous.

    :param n: the number of variables
    :type n: int
    :param parameters: the domain offset, then the lower capacities, then the upper capacities
    :type parameters: Sequence[int]
    :param domains: the initial domains, unused here
    :type domains: Sequence[tuple[int, int]]

    :return: True when no assignment can violate the constraint
    :rtype: bool
    """
    m = (len(parameters) - 1) >> 1  # number of values
    for j in range(m):
        if parameters[1 + m + j] < n:
            return False
    for j in range(m):
        if parameters[1 + j] != 0:
            return False
    return True


@njit(cache=True)
def _is_entailed(domains: NDArray, parameters: NDArray, m: int, possible: NDArray, fixed: NDArray) -> bool:
    """
    Returns whether every assignment within the domains satisfies the constraint: for each value, the variables
    that can take it are within its upper capacity and those fixed to it already meet its lower one. Both counts
    only move the safe way as domains shrink, so entailment holds for the rest of the subtree. Values outside
    the cover are not constrained, and are not counted.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: the first value, then the m lower capacities, then the m upper capacities
    :type parameters: NDArray
    :param m: the number of values
    :type m: int
    :param possible: scratch of m + 1 cells, a difference array of the variables that can take each value
    :type possible: NDArray
    :param fixed: scratch of m cells, the variables fixed to each value
    :type fixed: NDArray

    :return: True when the constraint is entailed
    :rtype: bool
    """
    first_value = parameters[0]
    # the counts add up to the domains' total width in the cover, so a total above the summed upper capacities rules
    # entailment out: a read-only pass that spares the counting when, as usual, the capacities are tight
    width = 0
    for i in range(len(domains)):
        lo = max(domains[i, DOMAIN_MIN], first_value)
        hi = min(domains[i, DOMAIN_MAX], first_value + m - 1)
        if lo <= hi:
            width += hi - lo + 1
    capacity = 0
    for v in range(m):
        capacity += parameters[1 + m + v]
    if width > capacity:
        return False
    for v in range(m + 1):
        possible[v] = 0
    for v in range(m):
        fixed[v] = 0
    for i in range(len(domains)):
        lo = max(domains[i, DOMAIN_MIN], first_value) - first_value
        hi = min(domains[i, DOMAIN_MAX], first_value + m - 1) - first_value
        if lo <= hi:
            possible[lo] += 1
            possible[hi + 1] -= 1
            if domains[i, DOMAIN_MIN] == domains[i, DOMAIN_MAX]:
                fixed[lo] += 1
    count = 0
    for v in range(m):
        count += possible[v]
        if count > parameters[1 + m + v] or fixed[v] < parameters[1 + v]:
            return False
    return True


@njit(cache=True)
def compute_domains_gcc(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    r"""
    This propagator (Global Cardinality Constraint) enforces that
    :math:`l_j \le |\{ i : x_i = v_j \}| \le c_j` for all j.
    It is adapted from "An efficient bounds consistency algorithm for the global cardinality constraint".

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: there are 1 + 2 * m parameters:
                       the first domain value (v_0), then the m lower bounds, then the m upper bounds (capacities)
    :type parameters: NDArray
    :param prop_state: this propagator's state block: [flag, l, u, bounds, t, d, h, min_sorted_vars,
                       max_sorted_vars, ranks, stable_intervals, stable_sets, new_mins],
                       sized by get_state_gcc
    :type prop_state: NDArray
    :return: a propagation status (PROP_INCONSISTENCY or PROP_CONSISTENCY)
    :rtype: int
    """
    n = len(domains)
    m = (len(parameters) - 1) >> 1  # number of values
    bounds_nb = 2 * (n + 1)
    psum_nb = 2 * (m + 6)
    # cell 0 is the change report the engine pre-sets and reads back; this propagator's own state follows
    psum_buffer = prop_state[2 : 2 + 2 * psum_nb]
    l = psum_buffer[:psum_nb].reshape(2, m + 6)
    u = psum_buffer[psum_nb:].reshape(2, m + 6)
    scratch = prop_state[2 + 2 * psum_nb :]
    bounds = scratch[:bounds_nb]
    t = scratch[bounds_nb : 2 * bounds_nb]  # critical capacity pointers
    d = scratch[2 * bounds_nb : 3 * bounds_nb]  # differences between critical capacities
    h = scratch[3 * bounds_nb : 4 * bounds_nb]  # Hall interval pointers
    min_sorted_vars = scratch[4 * bounds_nb : 4 * bounds_nb + n]
    max_sorted_vars = scratch[4 * bounds_nb + n : 4 * bounds_nb + 2 * n]
    ranks = scratch[4 * bounds_nb + 2 * n : 4 * bounds_nb + 4 * n].reshape(n, 2)
    zero_start = 4 * bounds_nb + 4 * n
    stable_intervals = scratch[zero_start : zero_start + bounds_nb]
    stable_sets = scratch[zero_start + bounds_nb : zero_start + 2 * bounds_nb]
    new_mins = scratch[zero_start + 2 * bounds_nb : zero_start + 2 * bounds_nb + n]
    counts_start = zero_start + 2 * bounds_nb + n
    possible = scratch[counts_start : counts_start + m + 1]
    fixed = scratch[counts_start + m + 1 :]
    # these three used to come from a fresh np.zeros every call; the persistent block needs the same
    # re-zeroing done explicitly, since it is no longer implied by a fresh allocation.
    #
    # Two of the three are wider than they have to be, and the third is not, which is worth writing down
    # because the asymmetry is not visible from here. filter_lower_gcc writes stable_intervals and
    # stable_sets over [1, nb + 1] before it reads either, and nothing in the algorithm reaches above
    # nb + 1: every value the two hold is a bound index no greater than that, path_max only ascends
    # through them, and path_set only walks where path_max can reach. Index 0 is the one cell that is
    # reachable and never written -- path_max is entered at stable_sets[y], which is 0 for the
    # lowest-ranked bound -- so for those two, `[0] = 0` is the whole of what the fill is doing. new_mins
    # gets no such argument: it is written inside one branch and read inside another, so whether a stale
    # cell can be read at all rests on those two conditions agreeing, which is the algorithm's invariant
    # and not something this code states.
    #
    # The narrow version was written and measured, and measures nothing: statistics identical on five gcc
    # models, and 0.7-1.2% on sports(8), against a model that drifts 10.6% run to run on this machine.
    # bounds_nb is 29 there, so all three arrays are ~284 bytes and never leave L1 however they are
    # filled -- the same reason every array merge and shrink tried here has measured zero. It was left
    # alone rather than landed: it trades a fill that is correct whatever the index ranges do for a
    # reachability argument that nothing checks, and buys nothing for it.
    prop_state[0] = 0  # raised at each write below: the engine skips the write-back scan while it stays 0
    if _is_entailed(domains, parameters, m, possible, fixed):
        return PROP_ENTAILMENT
    cold = prop_state[1] == 0  # the block is zeroed at solver init, so 0 means never called on this block
    if cold:
        prop_state[1] = 1
        # l and u are a function of parameters, which the engine never writes, so they are built once here
        # rather than on every call. init_partial_sum_into walks the ds row instead of filling it, so the
        # table has to start zeroed, exactly as the np.zeros this replaced left it.
        psum_buffer.fill(0)
        init_partial_sum_into(l, parameters[0], m, parameters[1 : 1 + m])
        init_partial_sum_into(u, parameters[0], m, parameters[1 + m :])
    while True:
        stable_intervals.fill(0)
        stable_sets.fill(0)
        new_mins.fill(0)
        if cold:
            argsort_into(min_sorted_vars, domains, DOMAIN_MIN)
            argsort_into(max_sorted_vars, domains, DOMAIN_MAX)
            cold = False
        else:
            argsort_into_warm(min_sorted_vars, domains, DOMAIN_MIN)
            argsort_into_warm(max_sorted_vars, domains, DOMAIN_MAX)
        nb = update_bounds(bounds, n, domains, ranks, min_sorted_vars, max_sorted_vars, l, u)
        # assert get_min_value(l) == get_min_value(u)
        # assert get_max_value(l) == get_max_value(u)
        # assert get_min_value(l) <= domains[min_sorted_vars[0], DOMAIN_MIN]
        # assert domains[max_sorted_vars[n - 1], DOMAIN_MAX] <= get_max_value(u)
        if get_sum(l, get_min_value(l), domains[min_sorted_vars[0], DOMAIN_MIN] - 1) > 0:
            return PROP_INCONSISTENCY
        if get_sum(l, domains[max_sorted_vars[n - 1], DOMAIN_MAX] + 1, get_max_value(l)) > 0:
            return PROP_INCONSISTENCY
        if not filter_lower_max(n, nb, t, d, h, bounds, domains, ranks, max_sorted_vars, u, prop_state):
            return PROP_INCONSISTENCY
        if not filter_lower_min(
            n,
            nb,
            t,
            d,
            h,
            bounds,
            domains,
            ranks,
            max_sorted_vars,
            l,
            stable_intervals,
            stable_sets,
            new_mins,
            prop_state,
        ):
            return PROP_INCONSISTENCY
        if not filter_upper_max(n, nb, t, d, h, bounds, domains, ranks, min_sorted_vars, u, prop_state):
            return PROP_INCONSISTENCY
        if not filter_upper_min(
            n, nb, t, d, h, bounds, domains, ranks, min_sorted_vars, l, stable_intervals, new_mins, prop_state
        ):
            return PROP_INCONSISTENCY
        # The passes can leave a bound on a value of capacity 0, which only the Hall reasoning of another call
        # would move off: move it here, and run the passes again when that moved one, so that a call is its own
        # fixpoint. Without a capacity of 0 this never moves anything.
        moved = skip_zero_capacities(domains, parameters, m)
        if moved < 0:
            return PROP_INCONSISTENCY
        if moved == 0:
            return PROP_CONSISTENCY
        prop_state[0] = 1


@njit(cache=True)
def skip_zero_capacities(domains: NDArray, parameters: NDArray, m: int) -> int:
    """
    Moves every bound off the values whose capacity (upper bound) is 0.

    :param domains: the domains of the variables
    :type domains: NDArray
    :param parameters: the first value, then the m lower bounds, then the m capacities
    :type parameters: NDArray
    :param m: the number of values
    :type m: int

    :return: -1 when a domain is left empty, 1 when a bound moved, 0 otherwise
    :rtype: int
    """
    first_value = parameters[0]
    capacities = parameters[1 + m :]
    moved = 0
    for i in range(len(domains)):
        lo = domains[i, DOMAIN_MIN]
        hi = domains[i, DOMAIN_MAX]
        while lo <= hi and first_value <= lo < first_value + m and capacities[lo - first_value] == 0:
            lo += 1
        while lo <= hi and first_value <= hi < first_value + m and capacities[hi - first_value] == 0:
            hi -= 1
        if lo > hi:
            return -1
        if lo != domains[i, DOMAIN_MIN] or hi != domains[i, DOMAIN_MAX]:
            domains[i, DOMAIN_MIN] = lo
            domains[i, DOMAIN_MAX] = hi
            moved = 1
    return moved
