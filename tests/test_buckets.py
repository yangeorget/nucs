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

from nucs.buckets import buckets_add, buckets_init, buckets_pop


class TestBuckets:
    def test_buckets(self) -> None:
        heap = buckets_init(4)
        # Bucket indices (already-bucketed weights). Lower bucket = higher priority.
        weights = np.array([0, 1, 2, 3])
        buckets_add(heap, 3, weights)
        buckets_add(heap, 2, weights)
        buckets_add(heap, 1, weights)
        buckets_add(heap, 0, weights)
        # Re-adding existing elements is a no-op (set semantics).
        buckets_add(heap, 3, weights)
        buckets_add(heap, 2, weights)
        buckets_add(heap, 1, weights)
        buckets_add(heap, 0, weights)
        assert buckets_pop(heap) == 0
        assert buckets_pop(heap) == 1
        assert buckets_pop(heap) == 2
        assert buckets_pop(heap) == 3
        assert buckets_pop(heap) == -1
