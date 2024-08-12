from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

START = 0
END = 1

MIN = 0
MAX = 1


def init_domains_by_values(domains: List[Union[int, Tuple[int, int]]]) -> NDArray:
    return np.array(
        [(domain, domain) if isinstance(domain, int) else domain for domain in domains],
        dtype=np.int32,
        # order="F",
    )


def init_indices_by_values(dom_indices: List[int]) -> NDArray:
    return np.array(dom_indices, dtype=np.uint16, order="C")


def init_indices(n: int) -> NDArray:
    return np.empty(n, dtype=np.uint16, order="C")


def init_domain_offsets_by_values(dom_offsets: List[int]) -> NDArray:
    return np.array(dom_offsets, dtype=np.int32, order="C")


def init_offsets(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32, order="C")


def init_data_by_values(data: List[int]) -> NDArray:
    return np.array(data, dtype=np.int32, order="C")


def init_data(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32, order="C")


def init_domain_changes(n: int, init_value: bool) -> NDArray:
    if init_value:
        return np.ones((n, 2), dtype=bool, order="F")
    else:
        return np.zeros((n, 2), dtype=bool, order="F")


def init_triggers(n: int, init_value: bool) -> NDArray:
    if init_value:
        return np.ones((n, 2), dtype=bool, order="F")
    else:
        return np.zeros((n, 2), dtype=bool, order="F")


def init_queue(n: int) -> NDArray:
    return np.empty(n, dtype=np.bool, order="C")


def init_algorithms(n: int) -> NDArray:
    return np.empty(n, dtype=np.int8, order="C")


def init_bounds(n: int) -> NDArray:
    return np.empty((n, 2), dtype=np.uint16)
