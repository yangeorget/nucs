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
import numpy as np

from nucs.heaps import min_heap_add, min_heap_init, min_heap_pop


class TestHeaps:
    def test_min_heap(self) -> None:
        heap = min_heap_init(4)
        weights = np.array([0, 1, 2, 3])
        min_heap_add(heap, 3, weights)
        min_heap_add(heap, 2, weights)
        min_heap_add(heap, 1, weights)
        min_heap_add(heap, 0, weights)
        min_heap_add(heap, 3, weights)
        min_heap_add(heap, 2, weights)
        min_heap_add(heap, 1, weights)
        min_heap_add(heap, 0, weights)
        assert heap[-1] == 4
        assert min_heap_pop(heap, weights) == 0
        assert heap[-1] == 3
        assert min_heap_pop(heap, weights) == 1
        assert heap[-1] == 2
        assert min_heap_pop(heap, weights) == 2
        assert heap[-1] == 1
        assert min_heap_pop(heap, weights) == 3
        assert heap[-1] == 0
        assert min_heap_pop(heap, weights) == -1
