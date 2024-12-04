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
import argparse
import sys

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import LOG_LEVEL_INFO, LOG_LEVELS, MAX, MIN
from nucs.examples.tsp.tsp_instances import GR17
from nucs.examples.tsp.tsp_problem import TSPProblem
from nucs.heuristics.heuristics import register_dom_heuristic, register_var_heuristic
from nucs.heuristics.value_dom_heuristic import value_dom_heuristic
from nucs.solvers.backtrack_solver import BacktrackSolver


@njit(cache=True)
def max_regret_var_heuristic(decision_domains: NDArray, shr_domains_stack: NDArray, stacks_top: NDArray) -> int:
    """
    :param shr_domains_stack: the stack of shared domains
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :return: the index of the shared domain
    """
    parameters = [
        [0, 633, 257, 91, 412, 150, 80, 134, 259, 505, 353, 324, 70, 211, 268, 246, 121],
        [633, 0, 390, 661, 227, 488, 572, 530, 555, 289, 282, 638, 567, 466, 420, 745, 518],
        [257, 390, 0, 228, 169, 112, 196, 154, 372, 262, 110, 437, 191, 74, 53, 472, 142],
        [91, 661, 228, 0, 383, 120, 77, 105, 175, 476, 324, 240, 27, 182, 239, 237, 84],
        [412, 227, 169, 383, 0, 267, 351, 309, 338, 196, 61, 421, 346, 243, 199, 528, 297],
        [150, 488, 112, 120, 267, 0, 63, 34, 264, 360, 208, 329, 83, 105, 123, 364, 35],
        [80, 572, 196, 77, 351, 63, 0, 29, 232, 444, 292, 297, 47, 150, 207, 332, 29],
        [134, 530, 154, 105, 309, 34, 29, 0, 249, 402, 250, 314, 68, 108, 165, 349, 36],
        [259, 555, 372, 175, 338, 264, 232, 249, 0, 495, 352, 95, 189, 326, 383, 202, 236],
        [505, 289, 262, 476, 196, 360, 444, 402, 495, 0, 154, 578, 439, 336, 240, 685, 390],
        [353, 282, 110, 324, 61, 208, 292, 250, 352, 154, 0, 435, 287, 184, 140, 542, 238],
        [324, 638, 437, 240, 421, 329, 297, 314, 95, 578, 435, 0, 254, 391, 448, 157, 301],
        [70, 567, 191, 27, 346, 83, 47, 68, 189, 439, 287, 254, 0, 145, 202, 289, 55],
        [211, 466, 74, 182, 243, 105, 150, 108, 326, 336, 184, 391, 145, 0, 57, 426, 96],
        [268, 420, 53, 239, 199, 123, 207, 165, 383, 240, 140, 448, 202, 57, 0, 483, 153],
        [246, 745, 472, 237, 528, 364, 332, 349, 202, 685, 542, 157, 289, 426, 483, 0, 336],
        [121, 518, 142, 84, 297, 35, 29, 36, 236, 390, 238, 301, 55, 96, 153, 336, 0],
    ]
    max_regret = 0
    best_idx = -1
    cp_top_idx = stacks_top[0]
    for dom_idx in decision_domains:
        shr_domain = shr_domains_stack[cp_top_idx, dom_idx]
        size = shr_domain[MAX] - shr_domain[MIN]  # actually this is size - 1
        if 0 < size:
            best_cost = sys.maxsize
            second_cost = sys.maxsize
            for value in range(shr_domain[MIN], shr_domain[MAX] + 1):
                cost = parameters[dom_idx][value]
                if cost > 0:
                    if cost < best_cost:
                        second_cost = best_cost
                        best_cost = cost
                    elif cost < second_cost:
                        second_cost = cost
            regret = second_cost - best_cost
            if max_regret < regret:
                best_idx = dom_idx
                max_regret = regret
    return best_idx


@njit(cache=True)
def min_cost_dom_heuristic(
    shr_domains_stack: NDArray,
    not_entailed_propagators_stack: NDArray,
    dom_update_stack: NDArray,
    stacks_top: NDArray,
    dom_idx: int,
) -> int:
    """
    :param shr_domains_stack: the stack of shared domains
    :param dom_update_stack: the stack of domain updates
    :param stacks_top: the index of the top of the stacks as a Numpy array
    :param dom_idx: the index of the shared domain
    :return: the bounds which are modified
    """
    parameters = [
        [0, 633, 257, 91, 412, 150, 80, 134, 259, 505, 353, 324, 70, 211, 268, 246, 121],
        [633, 0, 390, 661, 227, 488, 572, 530, 555, 289, 282, 638, 567, 466, 420, 745, 518],
        [257, 390, 0, 228, 169, 112, 196, 154, 372, 262, 110, 437, 191, 74, 53, 472, 142],
        [91, 661, 228, 0, 383, 120, 77, 105, 175, 476, 324, 240, 27, 182, 239, 237, 84],
        [412, 227, 169, 383, 0, 267, 351, 309, 338, 196, 61, 421, 346, 243, 199, 528, 297],
        [150, 488, 112, 120, 267, 0, 63, 34, 264, 360, 208, 329, 83, 105, 123, 364, 35],
        [80, 572, 196, 77, 351, 63, 0, 29, 232, 444, 292, 297, 47, 150, 207, 332, 29],
        [134, 530, 154, 105, 309, 34, 29, 0, 249, 402, 250, 314, 68, 108, 165, 349, 36],
        [259, 555, 372, 175, 338, 264, 232, 249, 0, 495, 352, 95, 189, 326, 383, 202, 236],
        [505, 289, 262, 476, 196, 360, 444, 402, 495, 0, 154, 578, 439, 336, 240, 685, 390],
        [353, 282, 110, 324, 61, 208, 292, 250, 352, 154, 0, 435, 287, 184, 140, 542, 238],
        [324, 638, 437, 240, 421, 329, 297, 314, 95, 578, 435, 0, 254, 391, 448, 157, 301],
        [70, 567, 191, 27, 346, 83, 47, 68, 189, 439, 287, 254, 0, 145, 202, 289, 55],
        [211, 466, 74, 182, 243, 105, 150, 108, 326, 336, 184, 391, 145, 0, 57, 426, 96],
        [268, 420, 53, 239, 199, 123, 207, 165, 383, 240, 140, 448, 202, 57, 0, 483, 153],
        [246, 745, 472, 237, 528, 364, 332, 349, 202, 685, 542, 157, 289, 426, 483, 0, 336],
        [121, 518, 142, 84, 297, 35, 29, 36, 236, 390, 238, 301, 55, 96, 153, 336, 0],
    ]
    cp_top_idx = stacks_top[0]
    best_cost = sys.maxsize
    best_value = -1
    shr_domain = shr_domains_stack[cp_top_idx, dom_idx]
    for value in range(shr_domain[MIN], shr_domain[MAX] + 1):
        cost = parameters[dom_idx][value]
        if 0 < cost < best_cost:
            best_cost = cost
            best_value = value
    value = value_dom_heuristic(
        shr_domains_stack, not_entailed_propagators_stack, dom_update_stack, stacks_top, dom_idx, best_value
    )
    # print(f"{dom_idx}={value}")
    return value


VAR_HEURISTIC_MAX_REGRET = register_var_heuristic(max_regret_var_heuristic)
DOM_HEURISTIC_MIN_COST = register_dom_heuristic(min_cost_dom_heuristic)


# Run with the following command (the second run is much faster because the code has been compiled):
# NUMBA_CACHE_DIR=.numba/cache python -m nucs.examples.alpha
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", choices=LOG_LEVELS, default=LOG_LEVEL_INFO)
    args = parser.parse_args()
    problem = TSPProblem(GR17)
    solver = BacktrackSolver(
        problem,
        decision_domains=list(range(len(GR17))),
        var_heuristic_idx=VAR_HEURISTIC_MAX_REGRET,
        dom_heuristic_idx=DOM_HEURISTIC_MIN_COST,
        log_level=args.log_level,
    )
    solution = solver.minimize(problem.shr_domain_nb - 1)
    print(solver.get_statistics())
