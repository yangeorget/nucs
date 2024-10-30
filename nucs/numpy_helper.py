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


def new_shr_domains_by_values(domains: List[Union[int, Tuple[int, int]]]) -> NDArray:
    """
    Only used in tests.
    """
    return np.array(
        [(domain, domain) if isinstance(domain, int) else domain for domain in domains], dtype=np.int32  # , order="F"
    )


def new_parameters_by_values(data: List[int]) -> NDArray:
    """
    Only used in tests.
    """
    return np.array(data, dtype=np.int32)
