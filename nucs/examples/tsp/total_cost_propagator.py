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
# Copyright 2024 - Yan Georget
###############################################################################
import sys

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, MAX, MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY


def get_complexity_total_cost(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return n * n


def get_triggers_total_cost(n: int, parameters: NDArray) -> NDArray:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    triggers = np.full(n, dtype=np.uint8, fill_value=EVENT_MASK_MIN_MAX)
    triggers[-1] = 0
    return triggers


@njit(cache=True)
def compute_domains_total_cost(domains: NDArray, parameters: NDArray) -> int:
    """
    :param domains: the domains of the variables
    :param parameters: the parameters of the propagator
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    n = len(domains) - 1
    used = np.zeros(n, dtype=np.bool)
    for i in range(n):
        if domains[i, MIN] == domains[i, MAX]:
            used[domains[i, MIN]] = True
    global_min = 0
    # for j in range(n):
    #     max_regret = 0
    #     min_cost = min([costs[i, j] for i in range(n) if costs[i,j] > 0])
    #     for i in range(n):
    #         if costs[i, j] == min_cost:
    #             regret = compute_regret(domains[i], costs[i])
    #             if regret > max_regret:
    #                 max_regret = regret
    #             global_min += regret
    #     global_min -= max_regret
    global_max = 0
    for i in range(n):
        if domains[i, MIN] == domains[i, MAX]:
            local_min = local_max = parameters[i * n + domains[i, MIN]]
        else:
            local_min = sys.maxsize
            local_max = 0
            for value in range(domains[i, MIN], domains[i, MAX] + 1):
                if not used[value]:
                    cost = parameters[i * n + value]
                    if cost > 0:
                        if cost < local_min:
                            local_min = cost
                        if cost > local_max:
                            local_max = cost
            # if local_min > local_max:
            #     return PROP_INCONSISTENCY
        global_min += local_min
        global_max += local_max
    domains[-1, MIN] = max(domains[-1, MIN], global_min)
    domains[-1, MAX] = min(domains[-1, MAX], global_max)
    if domains[-1, MIN] > domains[-1, MAX]:
        return PROP_INCONSISTENCY
    return PROP_CONSISTENCY
