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
from typing import Callable, Dict, List, Optional

from nucs.solvers.bound_consistency_algorithm import bound_consistency_algorithm

CONSISTENCY_ALG_FCTS: List[Callable] = []
CONSISTENCY_ALGS: Dict[str, int] = {}  # algorithm name to index, for name-based selection (eg from the CLI)


def register_consistency_algorithm(consistency_algorithm_fct: Callable, name: Optional[str] = None) -> int:
    """
    Register a consistency algorithm by adding its function to the corresponding list of functions.

    :param consistency_algorithm_fct: a function that enforces consistency
    :type consistency_algorithm_fct: Callable
    :param name: the name of the algorithm,
                 defaults to the function name without its _consistency_algorithm suffix, uppercased
    :type name: Optional[str]

    :return: the index of the consistency algorithm
    :rtype: int
    """
    CONSISTENCY_ALG_FCTS.append(consistency_algorithm_fct)
    if name is None:
        name = consistency_algorithm_fct.__name__.removesuffix("_consistency_algorithm").upper()
    CONSISTENCY_ALGS[name] = len(CONSISTENCY_ALG_FCTS) - 1
    return len(CONSISTENCY_ALG_FCTS) - 1


CONSISTENCY_ALG_BC = register_consistency_algorithm(bound_consistency_algorithm, "BC")
