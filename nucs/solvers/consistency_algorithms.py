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
from typing import Callable, List

from nucs.solvers.bound_consistency_algorithm import bound_consistency_algorithm
from nucs.solvers.shaving_consistency_algorithm import shaving_consistency_algorithm

CONSISTENCY_ALG_FCTS: List[Callable] = []


def register_consistency_algorithm(consistency_algorithm_fct: Callable) -> int:
    """
    Register a consistency algorithm by adding its function to the corresponding list of functions.

    :param consistency_algorithm_fct: a function that enforces consistency
    :type consistency_algorithm_fct: Callable

    :return: the index of the consistency algorithm
    :rtype: int
    """
    CONSISTENCY_ALG_FCTS.append(consistency_algorithm_fct)
    return len(CONSISTENCY_ALG_FCTS) - 1


CONSISTENCY_ALG_BC = register_consistency_algorithm(bound_consistency_algorithm)
CONSISTENCY_ALG_SHAVING = register_consistency_algorithm(shaving_consistency_algorithm)
