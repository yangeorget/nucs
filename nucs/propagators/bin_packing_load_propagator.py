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
import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY

# Budgets bounding the exact subset-sum reasoning so a single call stays cheap on large instances; beyond
# them the propagator falls back to the (always sound) load bounds and O(1) item rules.
SUBSET_SUM_CAP = 4096  # maximum total candidate weight for which the reachability array is built
ITEM_SUBSET_CAP = 48  # maximum number of candidates for which per-item no-sum pruning is run


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
def compute_domains_bin_packing_load(domains: NDArray, parameters: NDArray) -> int:
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

    :return: the status of the propagation (consistency or inconsistency) as an int
    :rtype: int
    """
    bin_low = parameters[0]
    item_nb = len(parameters) - 1
    bin_nb = domains.shape[0] - item_nb
    cand_idx = np.empty(item_nb, dtype=np.int64)
    cand_w = np.empty(item_nb, dtype=np.int64)
    for j in range(bin_nb):
        v = bin_low + j
        required = 0
        total = 0  # sum of the candidate weights
        nc = 0
        for i in range(item_nb):
            item = domains[bin_nb + i]
            if item[MIN] <= v <= item[MAX]:
                w = parameters[1 + i]
                if item[MIN] == item[MAX]:
                    required += w
                else:
                    cand_idx[nc] = i
                    cand_w[nc] = w
                    total += w
                    nc += 1
        load = domains[j]
        # load bounds: required <= load[j] <= required + (all candidates)
        load[MIN] = max(load[MIN], required)
        load[MAX] = min(load[MAX], required + total)
        if load[MIN] > load[MAX]:
            return PROP_INCONSISTENCY
        if nc == 0:
            continue  # the load is fully determined by the fixed items
        if total <= SUBSET_SUM_CAP:
            reach = _reach(cand_w, nc, -1, total)
            lo = load[MIN] - required
            hi = load[MAX] - required
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
            load[MIN] = max(load[MIN], required + min_c)
            load[MAX] = min(load[MAX], required + max_c)
            lo = load[MIN] - required
            hi = load[MAX] - required
            if nc <= ITEM_SUBSET_CAP:
                for t in range(nc):
                    item = domains[bin_nb + cand_idx[t]]
                    if item[MIN] == item[MAX]:
                        continue  # already committed by an earlier rule this pass
                    w = cand_w[t]
                    others = _reach(cand_w, nc, t, total - w)
                    # can the item still be in bin j? the others must fill [lo - w, hi - w]
                    if not _any_reachable(others, lo - w, hi - w):
                        if item[MIN] == v:
                            item[MIN] += 1
                        elif item[MAX] == v:
                            item[MAX] -= 1
                        if item[MIN] > item[MAX]:
                            return PROP_INCONSISTENCY
                        continue
                    # can the item still be out of bin j? the others alone must fill [lo, hi]
                    if not _any_reachable(others, lo, hi):
                        item[MIN] = v
                        item[MAX] = v
        else:
            # large candidate weight: fall back to the cheap sound rules
            for t in range(nc):
                item = domains[bin_nb + cand_idx[t]]
                if item[MIN] == item[MAX]:
                    continue
                w = cand_w[t]
                if required + w > load[MAX]:  # the item cannot fit in bin j
                    if item[MIN] == v:
                        item[MIN] += 1
                    elif item[MAX] == v:
                        item[MAX] -= 1
                    if item[MIN] > item[MAX]:
                        return PROP_INCONSISTENCY
                elif required + total - w < load[MIN]:  # nothing else can fill bin j
                    item[MIN] = v
                    item[MAX] = v
    # total-weight channeling: when every item is placed within range, the loads sum to the total weight
    weight_sum = 0
    all_in_range = True
    for i in range(item_nb):
        weight_sum += parameters[1 + i]
        item = domains[bin_nb + i]
        if item[MIN] < bin_low or item[MAX] > bin_low + bin_nb - 1:
            all_in_range = False
    if all_in_range:
        load_min_sum = 0
        load_max_sum = 0
        for j in range(bin_nb):
            load_min_sum += domains[j][MIN]
            load_max_sum += domains[j][MAX]
        for j in range(bin_nb):
            load = domains[j]
            new_min = weight_sum - (load_max_sum - load[MAX])
            new_max = weight_sum - (load_min_sum - load[MIN])
            load[MIN] = max(load[MIN], new_min)
            load[MAX] = min(load[MAX], new_max)
            if load[MIN] > load[MAX]:
                return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
