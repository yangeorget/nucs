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
# Copyright 2024-2025 - Yan Georget
###############################################################################
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_MIN_MAX, PROP_CONSISTENCY


def get_complexity_dummy(n: int, parameters: NDArray) -> float:
    """
    Returns the time complexity of the propagator as a float.
    :param n: the number of variables
    :param parameters: the parameters, unused here
    :return: a float
    """
    return 0.0


@njit(cache=True)
def get_triggers_dummy(n: int, dom_idx: int, parameters: NDArray) -> int:
    """
    This propagator is triggered whenever there is a change in the domain of a variable.
    :param parameters: the parameters, unused here
    :return: an array of triggers
    """
    return EVENT_MASK_MIN_MAX


@njit(cache=True)
def compute_domains_dummy(domains: NDArray, parameters: NDArray) -> int:
    """
    A propagator that does nothing.
    :return: the status of the propagation (consistency, inconsistency or entailment) as an int
    """
    return PROP_CONSISTENCY
