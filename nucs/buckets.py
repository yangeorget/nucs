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
from numba import njit  # type: ignore
from numpy.typing import NDArray

# Number of priority buckets.
# priorities[val] is interpreted directly as a bucket index; values >= NB_BUCKETS are clamped to the last bucket.
# Within a bucket: FIFO order (insertion order is preserved on pop).
BUCKET_NB = 8
BUCKET_FACTOR = 2


@njit(cache=True, fastmath=True)
def compute_priority(complexity: int) -> int:
    """
    Floor(log2(complexity)), clamped to [0, NB_BUCKETS-1].
    """
    if complexity <= 1:
        return 0
    bucket = 0
    while complexity > 1:
        complexity >>= BUCKET_FACTOR
        bucket += 1
    return min(bucket, BUCKET_NB - 1)


@njit(cache=True, fastmath=True)
def buckets_init(capacity: int) -> NDArray:
    """
    Creates a bucketed FIFO queue with set semantics over integers in [0, capacity).

    Layout (single int32 array):
      [0 : BUCKET_NB]                                          head per bucket (-1 = empty)
      [BUCKET_NB : 2*BUCKET_NB]                               tail per bucket (-1 = empty)
      [2*BUCKET_NB : 2*BUCKET_NB + capacity]                  next-pointer per element (-1 = tail)
      [2*BUCKET_NB + capacity : 2*BUCKET_NB + 2*capacity]     membership flag per element (0/1)
      [-1]                                                      cached min bucket index
    """
    return np.empty(2 * BUCKET_NB + 2 * capacity + 1, dtype=np.int32)


@njit(cache=True, fastmath=True)
def buckets_empty(buckets: NDArray, priorities: NDArray) -> None:
    """
    Empty the bucket FIFO queue.
    """
    nb = len(priorities)
    buckets[: 2 * BUCKET_NB + nb] = -1
    buckets[2 * BUCKET_NB + nb : 2 * BUCKET_NB + 2 * nb] = 0
    buckets[-1] = BUCKET_NB


@njit(cache=True, fastmath=True)
def buckets_add(buckets: NDArray, idx: int, priorities: NDArray) -> None:
    """
    Appends idx at the tail of bucket weights[idx].
    No-op if idx is already present.
    """
    capacity = (len(buckets) - 2 * BUCKET_NB - 1) >> 1
    membership_idx = 2 * BUCKET_NB + capacity + idx
    if buckets[membership_idx]:
        return
    bucket = priorities[idx]
    if bucket >= BUCKET_NB:
        bucket = BUCKET_NB - 1
    buckets[2 * BUCKET_NB + idx] = -1  # new tail has no successor
    old_tail = buckets[BUCKET_NB + bucket]
    if old_tail == -1:
        buckets[bucket] = idx  # bucket was empty, set head
    else:
        buckets[2 * BUCKET_NB + old_tail] = idx  # link previous tail to new node
    buckets[BUCKET_NB + bucket] = idx
    buckets[membership_idx] = 1
    if bucket < buckets[-1]:
        buckets[-1] = bucket


@njit(cache=True, fastmath=True)
def buckets_pop(buckets: NDArray) -> int:
    """
    Removes and returns the head of the lowest-priority non-empty bucket.
    :return: -1 if the queue is empty
    """
    capacity = (len(buckets) - 2 * BUCKET_NB - 1) >> 1  # TODO: pass a parameter?
    bucket = buckets[-1]
    while bucket < BUCKET_NB and buckets[bucket] == -1:
        bucket += 1
    if bucket == BUCKET_NB:
        buckets[-1] = BUCKET_NB
        return -1
    idx = buckets[bucket]
    new_head = buckets[2 * BUCKET_NB + idx]
    buckets[bucket] = new_head
    if new_head == -1:
        buckets[BUCKET_NB + bucket] = -1  # bucket now empty, clear tail too
    buckets[2 * BUCKET_NB + capacity + idx] = 0
    buckets[-1] = bucket
    return idx
