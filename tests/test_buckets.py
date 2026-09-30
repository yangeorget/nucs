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

from nucs.buckets import STORAGE_OFFSET, buckets_add, buckets_create, buckets_empty, buckets_init, buckets_pop


class TestBuckets:
    def test_buckets(self) -> None:
        buckets = buckets_create(4)
        # Bucket indices (already-bucketed weights). Lower bucket = higher priority.
        priorities = np.array([0, 1, 2, 3])
        membership_offset = STORAGE_OFFSET + 4
        buckets_empty(buckets, priorities)
        buckets_add(buckets, priorities, 3, membership_offset)
        buckets_add(buckets, priorities, 2, membership_offset)
        buckets_add(buckets, priorities, 1, membership_offset)
        buckets_add(buckets, priorities, 0, membership_offset)
        # Re-adding existing elements is a no-op (set semantics).
        buckets_add(buckets, priorities, 3, membership_offset)
        buckets_add(buckets, priorities, 2, membership_offset)
        buckets_add(buckets, priorities, 1, membership_offset)
        buckets_add(buckets, priorities, 0, membership_offset)
        assert buckets_pop(buckets, membership_offset) == 0
        assert buckets_pop(buckets, membership_offset) == 1
        assert buckets_pop(buckets, membership_offset) == 2
        assert buckets_pop(buckets, membership_offset) == 3
        assert buckets_pop(buckets, membership_offset) == -1

    def test_buckets_init_empties_the_queue_first(self) -> None:
        """A propagator left in the queue does not keep its old place: each bucket holds the propagators by index."""
        buckets = buckets_create(4)
        priorities = np.array([1, 0, 1, 0])
        membership_offset = STORAGE_OFFSET + 4
        buckets_empty(buckets, priorities)
        buckets_add(buckets, priorities, 2, membership_offset)
        buckets_add(buckets, priorities, 3, membership_offset)
        buckets_init(buckets, priorities)
        assert [buckets_pop(buckets, membership_offset) for _ in range(5)] == [1, 3, 0, 2, -1]
