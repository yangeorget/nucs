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

from nucs.problems.problem import BUCKET_NB

# the queue array starts with a head and a tail per bucket (see buckets_create)
STORAGE_OFFSET = BUCKET_NB << 1


@njit(cache=True)
def buckets_create(capacity: int) -> NDArray:
    """
    Creates a bucketed FIFO queue with set semantics over integers in [0, capacity).

    Layout (single int32 array):
      [0 : BUCKET_NB]                                              head per bucket (-1 = empty)
      [BUCKET_NB : STORAGE_OFFSET]                                 tail per bucket (-1 = empty)
      [STORAGE_OFFSET : STORAGE_OFFSET + capacity]                 next-pointer per element (-1 = tail)
      [STORAGE_OFFSET + capacity : STORAGE_OFFSET + 2*capacity]    membership flag per element (0/1)
      [-1]                                                         cached min bucket index
    """
    return np.empty(STORAGE_OFFSET + 2 * capacity + 1, dtype=np.int32)


@njit(cache=True)
def buckets_init(buckets: NDArray, priorities: NDArray) -> None:
    """
    Empties the queue, then adds every propagator to it, in the order of their indices within each bucket.

    The queue can hold anything before: garbage just after buckets_create, or the propagators that a failed descent
    left behind.

    :param buckets: the queue
    :type buckets: NDArray
    :param priorities: the bucket of each propagator
    :type priorities: NDArray
    """
    propagator_nb = len(priorities)
    buckets_empty(buckets, propagator_nb)
    membership_offset = STORAGE_OFFSET + propagator_nb
    for prop_idx in range(propagator_nb):
        buckets_add(buckets, priorities[prop_idx], prop_idx, membership_offset)


@njit(cache=True)
def buckets_empty(buckets: NDArray, propagator_nb: int) -> None:
    """
    Empties the bucket FIFO queue.

    :param buckets: the queue
    :type buckets: NDArray
    :param propagator_nb: the number of propagators, the capacity of the queue
    :type propagator_nb: int
    """
    buckets[: STORAGE_OFFSET + propagator_nb] = -1
    buckets[STORAGE_OFFSET + propagator_nb : STORAGE_OFFSET + 2 * propagator_nb] = 0
    buckets[-1] = BUCKET_NB


# always inlined, like buckets_pop: bc_algorithm is compiled without the reference-counting runtime, and a helper
# compiled on its own is inlined into it with the refcounts of whichever context first compiled it into the cache
@njit(cache=True, inline="always")
def buckets_add(buckets: NDArray, bucket: int, idx: int, membership_offset: int) -> None:
    """
    Appends idx at the tail of a bucket, and does nothing if idx is already in the queue.

    The caller gives the bucket, not an array of priorities: the propagation loop reads it from the INFO_PRIORITY
    column of propagator_infos, and the queue does not have to know that array.

    :param buckets: the queue
    :type buckets: NDArray
    :param bucket: the bucket of idx, clamped to the last bucket
    :type bucket: int
    :param idx: the element to add
    :type idx: int
    :param membership_offset: where the membership flags start in the queue, STORAGE_OFFSET + capacity
    :type membership_offset: int
    """
    membership_idx = membership_offset + idx
    if buckets[membership_idx]:
        return
    if bucket >= BUCKET_NB:
        bucket = BUCKET_NB - 1
    buckets[STORAGE_OFFSET + idx] = -1  # new tail has no successor
    old_tail = buckets[BUCKET_NB + bucket]
    if old_tail == -1:
        buckets[bucket] = idx  # bucket was empty, set head
    else:
        buckets[STORAGE_OFFSET + old_tail] = idx  # link previous tail to new node
    buckets[BUCKET_NB + bucket] = idx
    buckets[membership_idx] = 1
    buckets[-1] = min(buckets[-1], bucket)


@njit(cache=True, inline="always")
def buckets_pop(buckets: NDArray, membership_offset: int) -> int:
    """
    Removes and returns the head of the lowest-priority non-empty bucket.

    :return: -1 if the queue is empty
    :rtype: int
    """
    bucket = buckets[-1]
    while bucket < BUCKET_NB and buckets[bucket] == -1:
        bucket += 1
    if bucket == BUCKET_NB:
        buckets[-1] = BUCKET_NB
        return -1
    idx = buckets[bucket]
    new_head = buckets[STORAGE_OFFSET + idx]
    buckets[bucket] = new_head
    if new_head == -1:
        buckets[BUCKET_NB + bucket] = -1  # bucket now empty, clear tail too
    buckets[membership_offset + idx] = 0
    buckets[-1] = bucket
    return idx
