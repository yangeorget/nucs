from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def new_shr_domains_by_values(domains: List[Union[int, Tuple[int, int]]]) -> NDArray:
    return np.array(
        [(domain, domain) if isinstance(domain, int) else domain for domain in domains], dtype=np.int32, order="F"
    )


def new_dom_indices_by_values(dom_indices: List[int]) -> NDArray:
    return np.array(dom_indices, dtype=np.uint16)


def new_dom_indices(n: int) -> NDArray:
    return np.empty(n, dtype=np.uint16)


def new_dom_offsets_by_values(dom_offsets: List[int]) -> NDArray:
    return np.array(dom_offsets, dtype=np.int32)


def new_dom_offsets(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32)


def new_data_by_values(data: List[int]) -> NDArray:
    return np.array(data, dtype=np.int32)


def new_data(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32)


def new_triggers(n: int, init_value: bool) -> NDArray:
    if init_value:
        return np.ones((n, 2), dtype=bool)
    else:
        return np.zeros((n, 2), dtype=bool)


def new_triggered_propagators(n: int) -> NDArray:
    return np.ones(n, dtype=bool)


def new_not_entailed_propagators(n: int) -> NDArray:
    return np.ones(n, dtype=bool)


def new_bounds(n: int) -> NDArray:
    return np.zeros((n, 2), dtype=np.uint16)


def new_shr_domains_propagators(n: int, m: int) -> NDArray:
    return np.zeros((n, 2, m), dtype=bool)


def new_algorithms(n: int) -> NDArray:
    return np.empty(n, dtype=np.uint8)
