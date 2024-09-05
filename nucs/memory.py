from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

START = 0
END = 1

MIN = 0
MAX = 1

PROP_INCONSISTENCY = 0
PROP_CONSISTENCY = 1
PROP_ENTAILMENT = 2


def new_domains_by_values(domains: List[Union[int, Tuple[int, int]]]) -> NDArray:
    return np.array(
        [(domain, domain) if isinstance(domain, int) else domain for domain in domains],
        dtype=np.int32,
        order="F",
    )


def new_indices_by_values(dom_indices: List[int]) -> NDArray:
    return np.array(dom_indices, dtype=np.uint16, order="C")


def new_indices(n: int) -> NDArray:
    return np.empty(n, dtype=np.uint16, order="C")


def new_domain_offsets_by_values(dom_offsets: List[int]) -> NDArray:
    return np.array(dom_offsets, dtype=np.int32, order="C")


def new_offsets(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32, order="C")


def new_data_by_values(data: List[int]) -> NDArray:
    return np.array(data, dtype=np.int32, order="C")


def new_data(n: int) -> NDArray:
    return np.empty(n, dtype=np.int32, order="C")


def new_domain_changes(n: int) -> NDArray:
    return np.ones((n, 2), dtype=bool, order="F")


def new_triggers(n: int, init_value: bool) -> NDArray:
    if init_value:
        return np.ones((n, 2), dtype=bool, order="F")
    else:
        return np.zeros((n, 2), dtype=bool, order="F")


def new_triggered_propagators(n: int) -> NDArray:
    return np.empty(n, dtype=bool, order="C")


def new_entailed_propagators(n: int) -> NDArray:
    return np.zeros(n, dtype=bool, order="C")


def new_algorithms(n: int) -> NDArray:
    return np.empty(n, dtype=np.int8, order="C")


def new_bounds(n: int) -> NDArray:
    return np.empty((n, 2), dtype=np.uint16)


def new_propagators(n: int, m: int) -> NDArray:
    return np.zeros((n, 2, m), dtype=np.uint16, order="C")
