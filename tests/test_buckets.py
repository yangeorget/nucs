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

from nucs.buckets import buckets_add, buckets_init, buckets_pop, buckets_empty, BUCKET_NB


class TestBuckets:
    def test_buckets(self) -> None:
        buckets = buckets_init(4)
        # Bucket indices (already-bucketed weights). Lower bucket = higher priority.
        priorities = np.array([0, 1, 2, 3])
        storage_offset = BUCKET_NB << 1
        capacity = (len(buckets) - storage_offset - 1) >> 1
        membership_offset = storage_offset + capacity
        buckets_empty(buckets, priorities)
        buckets_add(buckets, priorities, 3, storage_offset, membership_offset)
        buckets_add(buckets, priorities, 2, storage_offset, membership_offset)
        buckets_add(buckets, priorities, 1, storage_offset, membership_offset)
        buckets_add(buckets, priorities, 0, storage_offset, membership_offset)
        # Re-adding existing elements is a no-op (set semantics).
        buckets_add(buckets, priorities, 3, storage_offset, membership_offset)
        buckets_add(buckets, priorities, 2, storage_offset, membership_offset)
        buckets_add(buckets, priorities, 1, storage_offset, membership_offset)
        buckets_add(buckets, priorities, 0, storage_offset, membership_offset)
        assert buckets_pop(buckets, storage_offset, membership_offset) == 0
        assert buckets_pop(buckets, storage_offset, membership_offset) == 1
        assert buckets_pop(buckets, storage_offset, membership_offset) == 2
        assert buckets_pop(buckets, storage_offset, membership_offset) == 3
        assert buckets_pop(buckets, storage_offset, membership_offset) == -1
