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
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from nucs.constants import STACK_MAX_HEIGHT


def new_not_entailed_propagators(n: int) -> NDArray:
    return np.ones(n, dtype=np.bool)


def new_shr_domains_by_values(domains: List[Union[int, Tuple[int, int]]]) -> NDArray:
    return np.array(
        [(domain, domain) if isinstance(domain, int) else domain for domain in domains], dtype=np.int32, order="F"
    )


def new_stacks(domains: List[Union[int, Tuple[int, int]]], propagator_nb: int) -> NDArray:
    shr_domains_stack = np.empty((STACK_MAX_HEIGHT, len(domains)), dtype=np.int32)
    shr_domains_stack[0] = new_shr_domains_by_values(domains)
    new_not_entailed_propagators_stack = np.empty((STACK_MAX_HEIGHT, propagator_nb), dtype=np.bool)
    new_not_entailed_propagators_stack[0] = new_not_entailed_propagators(propagator_nb)
    height = np.array((1,), dtype=np.uint8)
    height[0] = 1
    return shr_domains_stack, new_not_entailed_propagators_stack, height


def new_dom_indices_by_values(dom_indices: List[int]) -> NDArray:
    return np.array(dom_indices, dtype=np.uint16)


def new_dom_indices(n: int) -> NDArray:
    return np.empty(n, dtype=np.uint16)


def new_dom_offsets_by_values(dom_offsets: List[int]) -> NDArray:
    return np.array(dom_offsets, dtype=np.int32)


def new_dom_offsets(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32)


def new_parameters_by_values(data: List[int]) -> NDArray:
    return np.array(data, dtype=np.int32)


def new_parameters(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32)


def new_triggers(n: int, init_value: bool) -> NDArray:
    if init_value:
        return np.ones((n, 2), dtype=np.bool)
    else:
        return np.zeros((n, 2), dtype=np.bool)


def new_triggered_propagators(n: int) -> NDArray:
    return np.ones(n, dtype=np.bool)


def new_bounds(n: int) -> NDArray:
    return np.zeros((n, 2), dtype=np.uint16)


def new_shr_domains_propagators(n: int, m: int) -> NDArray:
    return np.zeros((n, 2, m), dtype=np.bool)


def new_algorithms(n: int) -> NDArray:
    return np.empty(n, dtype=np.uint8)
