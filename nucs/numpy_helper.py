####################################################################
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

import numpy as np
from numpy._typing import NDArray


def flatten_arrays(arrays: list[NDArray]) -> tuple[NDArray, NDArray]:
    """
    Flattens a ragged list of arrays into their concatenation and the CSR offsets delimiting each array.

    :param arrays: the arrays, of identical dtype but possibly different shapes
    :type arrays: List[NDArray]

    :return: the flat concatenation and the offsets
    :rtype: Tuple[NDArray, NDArray]
    """
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([array.size for array in arrays], out=offsets[1:])
    return np.concatenate([array.reshape(-1) for array in arrays]), offsets
