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
from collections.abc import Sequence

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_MIN_MAX, PROP_CONSISTENCY, PROP_INCONSISTENCY

# Budgets bounding the exact subset-sum reasoning so a single call stays cheap on large instances; beyond
# them the propagator falls back to the (always sound) load bounds and O(1) item rules.
SUBSET_SUM_CAP = 4096  # maximum total candidate weight for which the reachability array is built
ITEM_SUBSET_CAP = 48  # maximum number of candidates for which per-item no-sum pruning is run


STATE_SUF = 0  # the suffix reachability table, (nc_cap + 2) rows of (total_cap + 1)


def _table_shape(parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the (rows, columns) of the suffix reachability table this propagator can need.

    Both dimensions are already bounded by the caps the exact reasoning runs under, and the column count is
    bounded again by the weights actually posted, so a small instance reserves a small table.

    :param parameters: the bin offset followed by the item weights
    :type parameters: Sequence[int]

    :return: the number of rows and columns
    :rtype: tuple[int, int]
    """
    item_nb = len(parameters) - 1
    total_cap = min(sum(int(w) for w in parameters[1:]), SUBSET_SUM_CAP)
    return min(item_nb, ITEM_SUBSET_CAP) + 2, total_cap + 1


def get_state_bin_packing_load(n: int, parameters: Sequence[int]) -> tuple[int, int]:
    """
    Returns the size of this propagator's state block: the suffix reachability table the item rule needs.

    The item rule asks, for each candidate in turn, whether the *other* candidates can still fill the bin,
    and was answering it by rebuilding the whole subset-sum without that candidate -- O(nc^2 * total) for a
    bin, and on the model in datasets/fzn 1.67 billion inner iterations. Reachability without candidate t is
    the subsets of those before t combined with the subsets of those after it, so one pass from the right
    records every suffix and one running prefix covers the rest: O(nc * total) for the whole rule.

    The table is an untrailed hint, rebuilt from the domains on every call; it is state only in the sense
    that the memory persists. At the caps the exact reasoning runs under it is at most (48 + 2) * 4097
    cells, and usually far less, since the column count follows the weights actually posted.

    :param n: the number of variables, the loads followed by the items
    :type n: int
    :param parameters: the bin offset followed by the item weights
    :type parameters: Sequence[int]

    :return: (trailed_nb, hint_nb)
    :rtype: tuple[int, int]
    """
    rows, cols = _table_shape(parameters)
    return 0, rows * cols


def get_complexity_bin_packing_load(n: int, parameters: NDArray) -> int:
    """
    Returns the time complexity of the propagator as an int.

    :param n: the number of variables (loads and bins)
    :type n: int
    :param parameters: the bin offset followed by the item weights
    :type parameters: NDArray

    :return: an int
    :rtype: int
    """
    item_nb = len(parameters) - 1
    bin_nb = n - item_nb
    return item_nb * item_nb * bin_nb


@njit(cache=True)
def get_triggers_bin_packing_load(n: int, variable: int, parameters: NDArray) -> int:
    """
    Wakes on every bound change: bin bounds define the required/possible load and load bounds drive the
    reverse pruning of the bins.

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


@njit(cache=True)
def _reach(weights: NDArray, count: int, skip: int, total: int) -> NDArray:
    """
    Builds the subset-sum reachability of ``weights[0:count]`` (optionally skipping index ``skip``).

    ``result[s]`` is 1 iff some subset of the selected weights sums to ``s``.

    :param weights: the candidate weights
    :type weights: NDArray
    :param count: the number of candidates to consider
    :type count: int
    :param skip: an index to exclude, or -1 to include them all
    :type skip: int
    :param total: the sum of the selected weights (the size of the reachability array minus one)
    :type total: int

    :return: the reachability array of length total + 1
    :rtype: NDArray
    """
    reach = np.zeros(total + 1, dtype=np.uint8)
    reach[0] = 1
    for t in range(count):
        if t == skip:
            continue
        w = weights[t]
        for s in range(total, w - 1, -1):
            if reach[s - w]:
                reach[s] = 1
    return reach


@njit(cache=True)
def _any_split(prefix: NDArray, counts: NDArray, lo: int, hi: int, total: int, without_total: int) -> bool:
    """
    Returns whether some subset of the candidates other than one reaches a sum in ``[lo, hi]``.

    Those subsets are exactly a subset of the candidates before it plus one of those after, so the question
    is whether any reachable prefix sum ``a`` leaves ``[lo - a, hi - a]`` reachable in the suffix. ``counts``
    is the suffix's running count, which makes each of those tests a subtraction.

    :param prefix: reachability of the candidates before the excluded one, as 0/1 by sum
    :type prefix: NDArray
    :param counts: how many suffix sums are reachable up to and including each index
    :type counts: NDArray
    :param lo: the lowest sum that would do
    :type lo: int
    :param hi: the highest sum that would do
    :type hi: int
    :param total: the largest sum either side is indexed by
    :type total: int
    :param without_total: the largest sum reachable without the excluded candidate
    :type without_total: int

    :return: whether the two sides together reach the window
    :rtype: bool
    """
    if hi < 0 or lo > without_total:
        return False
    for a in range(min(total, without_total) + 1):
        if prefix[a]:
            b_lo = max(lo - a, 0)
            b_hi = min(hi - a, without_total)
            if b_lo > b_hi:
                continue
            if counts[b_hi] - (counts[b_lo - 1] if b_lo > 0 else 0) > 0:
                return True
    return False


@njit(cache=True)
def _any_reachable(reach: NDArray, lo: int, hi: int) -> bool:
    """
    Returns whether any value in ``[lo, hi]`` (clamped to the array) is reachable.

    :param reach: a reachability array
    :type reach: NDArray
    :param lo: the lower bound
    :type lo: int
    :param hi: the upper bound
    :type hi: int

    :return: whether some value in the range is reachable
    :rtype: bool
    """
    lo = max(lo, 0)
    hi = min(hi, len(reach) - 1)
    for s in range(lo, hi + 1):
        if reach[s]:
            return True
    return False


@njit(cache=True)
def compute_domains_bin_packing_load(domains: NDArray, parameters: NDArray, prop_state: NDArray) -> int:
    """
    Implements the bin_packing_load constraint: each item i (with non-negative weight w[i]) is placed in bin
    bin[i], and load[j] equals the sum of the weights of the items placed in bin j.

    The first ``bin_nb`` domains are the loads, the remaining ``item_nb`` domains are the bins; parameters[0]
    is the bin offset (the value of bin[i] denoting the first load) and parameters[1:] are the item weights.

    Filtering (weights are non-negative, per the MiniZinc contract), iterated to a fixpoint so a single call is
    idempotent:

    - each load[j] is bounded below by its required load (items fixed to j) and above by its possible load;
    - when the candidate weights are small enough, exact subset-sum reasoning (Shaw's no-sum, but complete)
      tightens load[j] to a value its candidates can actually reach, prunes a candidate from a bin when the
      remaining items cannot complete a valid load, and forces a candidate into a bin when nothing else can;
    - otherwise the cheap O(1) overflow / forced rules are used.

    :param domains: the domains of the loads then the bins
    :type domains: NDArray
    :param parameters: the bin offset then the item weights
    :type parameters: NDArray
    :param prop_state: this propagator's state block (unused)
    :type prop_state: NDArray

    :return: the status of the propagation (consistency or inconsistency) as an int
    :rtype: int
    """
    bin_low = parameters[0]
    item_nb = len(parameters) - 1
    bin_nb = domains.shape[0] - item_nb
    cand_idx = np.empty(item_nb, dtype=np.int64)
    cand_w = np.empty(item_nb, dtype=np.int64)
    cols_cap = len(prop_state) // (min(item_nb, ITEM_SUBSET_CAP) + 2)
    suffix = prop_state
    prefix = np.empty(cols_cap, dtype=np.int32)
    counts = np.empty(cols_cap, dtype=np.int32)
    for j in range(bin_nb):
        v = bin_low + j
        required = 0
        total = 0  # sum of the candidate weights
        nc = 0
        for i in range(item_nb):
            item = domains[bin_nb + i]
            if item[DOMAIN_MIN] <= v <= item[DOMAIN_MAX]:
                w = parameters[1 + i]
                if item[DOMAIN_MIN] == item[DOMAIN_MAX]:
                    required += w
                else:
                    cand_idx[nc] = i
                    cand_w[nc] = w
                    total += w
                    nc += 1
        load = domains[j]
        # load bounds: required <= load[j] <= required + (all candidates)
        load[DOMAIN_MIN] = max(load[DOMAIN_MIN], required)
        load[DOMAIN_MAX] = min(load[DOMAIN_MAX], required + total)
        if load[DOMAIN_MIN] > load[DOMAIN_MAX]:
            return PROP_INCONSISTENCY
        if nc == 0:
            continue  # the load is fully determined by the fixed items
        if total <= SUBSET_SUM_CAP:
            reach = _reach(cand_w, nc, -1, total)
            lo = load[DOMAIN_MIN] - required
            hi = load[DOMAIN_MAX] - required
            # tighten the load to the sub-range its candidates can actually sum to
            min_c = -1
            for s in range(lo, hi + 1):
                if reach[s]:
                    min_c = s
                    break
            if min_c < 0:
                return PROP_INCONSISTENCY
            max_c = -1
            for s in range(hi, lo - 1, -1):
                if reach[s]:
                    max_c = s
                    break
            load[DOMAIN_MIN] = max(load[DOMAIN_MIN], required + min_c)
            load[DOMAIN_MAX] = min(load[DOMAIN_MAX], required + max_c)
            lo = load[DOMAIN_MIN] - required
            hi = load[DOMAIN_MAX] - required
            if nc <= ITEM_SUBSET_CAP:
                # reachability without candidate t is the subsets before t combined with those after it.
                # suf[t] holds the subsets of candidates t..nc-1, built once from the right; prefix holds
                # the subsets of 0..t-1, carried forward as the loop advances. Between them every "without
                # t" question is answered without rebuilding anything.
                cols = total + 1
                for sm in range(cols):
                    suffix[nc * cols + sm] = 0
                suffix[nc * cols] = 1
                for t in range(nc - 1, -1, -1):
                    w = cand_w[t]
                    src = (t + 1) * cols
                    dst = t * cols
                    for sm in range(cols):
                        suffix[dst + sm] = suffix[src + sm]
                    for sm in range(total, w - 1, -1):
                        if suffix[src + sm - w]:
                            suffix[dst + sm] = 1
                for sm in range(cols):
                    prefix[sm] = 0
                prefix[0] = 1
                for t in range(nc):
                    item = domains[bin_nb + cand_idx[t]]
                    w = cand_w[t]
                    if item[DOMAIN_MIN] != item[DOMAIN_MAX]:  # not already committed by an earlier rule
                        # running count of the suffix, so "does it reach anything in [a, b]" is O(1)
                        src = (t + 1) * cols
                        running = 0
                        for sm in range(cols):
                            running += suffix[src + sm]
                            counts[sm] = running
                        without_total = total - w
                        # can the item still be in bin j? the others must fill [lo - w, hi - w]
                        if not _any_split(prefix, counts, lo - w, hi - w, total, without_total):
                            if item[DOMAIN_MIN] == v:
                                item[DOMAIN_MIN] += 1
                            elif item[DOMAIN_MAX] == v:
                                item[DOMAIN_MAX] -= 1
                            if item[DOMAIN_MIN] > item[DOMAIN_MAX]:
                                return PROP_INCONSISTENCY
                        # can the item still be out of bin j? the others alone must fill [lo, hi]
                        elif not _any_split(prefix, counts, lo, hi, total, without_total):
                            item[DOMAIN_MIN] = v
                            item[DOMAIN_MAX] = v
                    for sm in range(total, w - 1, -1):  # advance the prefix past candidate t
                        if prefix[sm - w]:
                            prefix[sm] = 1
        else:
            # large candidate weight: fall back to the cheap sound rules
            for t in range(nc):
                item = domains[bin_nb + cand_idx[t]]
                if item[DOMAIN_MIN] == item[DOMAIN_MAX]:
                    continue
                w = cand_w[t]
                if required + w > load[DOMAIN_MAX]:  # the item cannot fit in bin j
                    if item[DOMAIN_MIN] == v:
                        item[DOMAIN_MIN] += 1
                    elif item[DOMAIN_MAX] == v:
                        item[DOMAIN_MAX] -= 1
                    if item[DOMAIN_MIN] > item[DOMAIN_MAX]:
                        return PROP_INCONSISTENCY
                elif required + total - w < load[DOMAIN_MIN]:  # nothing else can fill bin j
                    item[DOMAIN_MIN] = v
                    item[DOMAIN_MAX] = v
    # total-weight channeling: when every item is placed within range, the loads sum to the total weight
    weight_sum = 0
    all_in_range = True
    for i in range(item_nb):
        weight_sum += parameters[1 + i]
        item = domains[bin_nb + i]
        if item[DOMAIN_MIN] < bin_low or item[DOMAIN_MAX] > bin_low + bin_nb - 1:
            all_in_range = False
    if all_in_range:
        load_min_sum = 0
        load_max_sum = 0
        for j in range(bin_nb):
            load_min_sum += domains[j][DOMAIN_MIN]
            load_max_sum += domains[j][DOMAIN_MAX]
        for j in range(bin_nb):
            load = domains[j]
            new_min = weight_sum - (load_max_sum - load[DOMAIN_MAX])
            new_max = weight_sum - (load_min_sum - load[DOMAIN_MIN])
            load[DOMAIN_MIN] = max(load[DOMAIN_MIN], new_min)
            load[DOMAIN_MAX] = min(load[DOMAIN_MAX], new_max)
            if load[DOMAIN_MIN] > load[DOMAIN_MAX]:
                return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
