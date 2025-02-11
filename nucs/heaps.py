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
# Copyright 2024-2025 - Yan Georget
###############################################################################


import numpy as np
from numba import njit
from numpy.typing import NDArray


def min_heap_init(capacity: int) -> NDArray:
    """
    Creates a data structure, designed to store integers from 0 to n-1,
    which is both a min-heap and a set (an element cannot be added twice).
    This data structure is implemented with a Numpy array of size 2n+1:
    the n first slots store the heap,
    the n following slots store the set,
    the last slot is the size of the heap.
    """
    return np.zeros(2 * capacity + 1, dtype=np.uint32)


@njit(cache=True)
def min_heap_add(heap: NDArray, capacity: int, val: int) -> None:
    if heap[capacity + val] == 0:
        heap[heap[-1]] = val
        min_heap_up(heap, heap[-1])
        heap[-1] += 1
        heap[capacity + val] = 1


@njit(cache=True)
def min_heap_pop(heap: NDArray, capacity: int) -> int:
    if heap[-1] == 0:
        return -1
    heap[-1] -= 1
    val = heap[0]
    min_heap_swap(heap, 0, heap[-1])
    min_heap_down(heap, 0)
    heap[capacity + val] = 0
    return val


@njit(cache=True)
def min_heap_swap(heap: NDArray, i: int, j: int) -> None:
    tmp = heap[i]
    heap[i] = heap[j]
    heap[j] = tmp


@njit(cache=True)
def min_heap_up(heap: NDArray, pos: int) -> None:
    """
    No recursive version because of a Numba bug.
    """
    while pos > 0:
        father = (pos - 1) >> 1
        if heap[father] <= heap[pos]:
            break
        min_heap_swap(heap, pos, father)
        pos = father


@njit(cache=True)
def min_heap_down(heap: NDArray, pos: int) -> None:
    """
    No recursive version because of a Numba bug.
    """
    while True:
        left = (pos << 1) + 1
        if left >= heap[-1]:
            break
        right = left + 1
        smallest = right if right < heap[-1] and heap[right] < heap[left] else left
        if heap[pos] <= heap[smallest]:
            break
        min_heap_swap(heap, pos, smallest)
        pos = smallest
