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
import sys

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import MAX, MIN


@njit(cache=True, fastmath=True)
def critical_resource_var_heuristic(
    decision_variables: NDArray, domains_stk: NDArray, top: int, params: NDArray
) -> int:
    """
    Chooses a task to branch on for disjunctive (unary resource) scheduling problems such as the job-shop.

    Each decision variable is the start time of a task; ``params[v]`` carries ``(resource, duration)`` of that
    task. Rather than picking a globally smallest domain -- which hops from one resource to another -- this
    heuristic focuses branching on a single *critical* resource until its tasks are sequenced, mirroring the
    "keep the critical machine until its tasks are ordered" strategy: the search stays on one resource so the
    disjunctive propagator can sequence it, before moving on.

    The critical resource is the one (with an unbound task) of smallest slack ``max(lct) - min(est) - load``
    -- the least free time, hence the most likely to fail -- breaking ties on the tightest start window. Within
    it the tightest unbound task is selected (smallest window, then smallest earliest start), to be split low.

    :param decision_variables: the decision variables, the start times of the tasks
    :type decision_variables: NDArray
    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param top: the index of the top of the stacks
    :type top: int
    :param params: a two-dimensional array, ``params[v]`` is ``(resource, duration)`` of task ``v``
    :type params: NDArray

    :return: the variable, or -1 when every resource is sequenced (all start times bound)
    :rtype: int
    """
    inf = sys.maxsize
    resource_nb = 0
    for variable in decision_variables:
        resource = params[variable, 0]
        if resource >= resource_nb:
            resource_nb = resource + 1
    if resource_nb <= 0:
        return -1
    est_min = np.full(resource_nb, inf, dtype=np.int64)
    lct_max = np.full(resource_nb, -inf, dtype=np.int64)
    load = np.zeros(resource_nb, dtype=np.int64)
    tightest = np.full(resource_nb, inf, dtype=np.int64)  # smallest unbound start window, inf when none
    for variable in decision_variables:
        resource = params[variable, 0]
        if resource < 0:
            continue
        lo = domains_stk[top, variable, MIN]
        hi = domains_stk[top, variable, MAX]
        duration = params[variable, 1]
        if lo < est_min[resource]:
            est_min[resource] = lo
        completion = hi + duration
        if completion > lct_max[resource]:
            lct_max[resource] = completion
        load[resource] += duration
        if lo < hi and hi - lo < tightest[resource]:
            tightest[resource] = hi - lo
    # the critical resource: smallest slack among resources that still have an unbound task
    critical = -1
    best_slack = inf
    best_tightest = inf
    for resource in range(resource_nb):
        if tightest[resource] < inf:  # has at least one unbound task
            slack = lct_max[resource] - est_min[resource] - load[resource]
            if slack < best_slack or (slack == best_slack and tightest[resource] < best_tightest):
                best_slack = slack
                best_tightest = tightest[resource]
                critical = resource
    if critical == -1:
        return -1
    # the tightest unbound task on the critical resource
    best_variable = -1
    best_width = inf
    best_est = inf
    for variable in decision_variables:
        if params[variable, 0] != critical:
            continue
        lo = domains_stk[top, variable, MIN]
        hi = domains_stk[top, variable, MAX]
        if lo < hi:
            width = hi - lo
            if width < best_width or (width == best_width and lo < best_est):
                best_width = width
                best_est = lo
                best_variable = variable
    return best_variable
