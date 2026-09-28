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
"""
The restart policies: after how many failures each descent of the search stops and restarts from the root.

The names and the parameters are those of the MiniZinc restart annotations. A limit counts failures, as Gecode's
cutoffs do. The linear, geometric and Luby limits have no upper bound, so a descent eventually becomes long enough
to finish the search: these restarts keep the search complete. A constant limit does not.
"""

import itertools
from collections.abc import Iterator

RESTART_NONE = "none"
RESTART_CONSTANT = "constant"  # scale, scale, scale, ...
RESTART_LINEAR = "linear"  # scale, 2 scale, 3 scale, ...
RESTART_GEOMETRIC = "geometric"  # scale, scale base, scale base^2, ...
RESTART_LUBY = "luby"  # scale times the Luby sequence 1, 1, 2, 1, 1, 2, 4, ...
RESTART_POLICIES = (RESTART_NONE, RESTART_CONSTANT, RESTART_LINEAR, RESTART_GEOMETRIC, RESTART_LUBY)


def luby(i: int) -> int:
    """
    Returns the i-th term of the Luby sequence 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...

    :param i: the index of the term, from 1
    :type i: int

    :return: the term
    :rtype: int
    """
    while True:
        k = i.bit_length()
        if i == (1 << k) - 1:  # the last term of a block of length 2^k - 1
            return 1 << (k - 1)
        i -= (1 << (k - 1)) - 1  # the same term in the previous block


def restart_limits(policy: str, scale: int, base: float = 2.0) -> Iterator[int]:
    """
    Returns the failure limits of the successive descents of the search.

    A constant limit is a restart policy too, but the only one that can make a search incomplete: it is accepted
    because MiniZinc defines it, and it is the model's decision.

    :param policy: one of RESTART_POLICIES
    :type policy: str
    :param scale: the number of failures the policy multiplies, at least 1
    :type scale: int
    :param base: the ratio of the geometric policy, above 1
    :type base: float

    :return: the limits, -1 for ever when the policy is RESTART_NONE
    :rtype: Iterator[int]
    """
    if policy not in RESTART_POLICIES:
        raise ValueError(f"Unknown restart policy {policy}, expected one of {RESTART_POLICIES}")
    if policy == RESTART_NONE:
        return itertools.repeat(-1)
    if scale < 1:
        raise ValueError(f"The restart scale must be at least 1, not {scale}")
    if policy == RESTART_CONSTANT:
        return itertools.repeat(scale)
    if policy == RESTART_LINEAR:
        return (scale * i for i in itertools.count(1))
    if policy == RESTART_GEOMETRIC:
        if base <= 1.0:
            raise ValueError(f"The geometric restart base must be above 1, not {base}")
        return (max(1, round(scale * base**i)) for i in itertools.count())
    return (scale * luby(i) for i in itertools.count(1))
