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
The bound literals of lazy clause generation (design/proposals/lcg.md, 1.2 and 3.2).

A literal is a fact about one bound of one variable: [x >= v] or [x <= v]. NuCS domains are intervals, so these two
kinds are sufficient, and no Boolean variable is created for a literal. A literal is the pair (var_side, value):

    var_side = (variable << 1) | side, with side = DOMAIN_MIN for [x >= value] and DOMAIN_MAX for [x <= value]

var_side is also the index in state of the bound that the literal is about, so a literal reads the domains directly.
A literal is true when the current bounds make it true, false when its negation is true, and unassigned otherwise.
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MIN

# The columns of an array of literals, one literal on each row.
LIT_VAR_SIDE = 0
LIT_VALUE = 1
LIT_WIDTH = 2


@njit(cache=True, inline="always")
def lit_negate(var_side: int, value: int) -> tuple[int, int]:
    """
    Returns the negation of a literal: [x >= v] gives [x <= v - 1], and [x <= v] gives [x >= v + 1].

    The negation is about the other bound of the same variable, so only the low bit of var_side changes. The value
    moves by -1 for a lower bound and by +1 for an upper bound, with no branch.

    :param var_side: the variable shifted left by one, with the side of the bound in the low bit
    :type var_side: int
    :param value: the value of the literal
    :type value: int

    :return: the var_side and the value of the negation
    :rtype: tuple[int, int]
    """
    return var_side ^ 1, value + ((var_side & 1) << 1) - 1


@njit(cache=True, inline="always")
def lit_is_true(state: NDArray, var_side: int, value: int) -> bool:
    """
    Returns whether the current bounds make a literal true.

    A bound makes true all the weaker literals on the same side: after [x >= 5], [x >= 4] is true too.

    :param state: all the backtrackable state, whose head holds the bounds of the domains
    :type state: NDArray
    :param var_side: the variable shifted left by one, with the side of the bound in the low bit
    :type var_side: int
    :param value: the value of the literal
    :type value: int

    :return: true if the literal is true
    :rtype: bool
    """
    if (var_side & 1) == DOMAIN_MIN:
        return state[var_side] >= value
    return state[var_side] <= value


@njit(cache=True, inline="always")
def lit_is_false(state: NDArray, var_side: int, value: int) -> bool:
    """
    Returns whether the current bounds make a literal false, that is, its negation true.

    The test reads the other bound of the variable, at var_side ^ 1: [x >= v] is false when the upper bound of x is
    below v, and [x <= v] is false when the lower bound of x is above v.

    :param state: all the backtrackable state, whose head holds the bounds of the domains
    :type state: NDArray
    :param var_side: the variable shifted left by one, with the side of the bound in the low bit
    :type var_side: int
    :param value: the value of the literal
    :type value: int

    :return: true if the literal is false
    :rtype: bool
    """
    if (var_side & 1) == DOMAIN_MIN:
        return state[var_side ^ 1] < value
    return state[var_side ^ 1] > value


@njit(cache=True, inline="always")
def lit_is_unassigned(state: NDArray, var_side: int, value: int) -> bool:
    """
    Returns whether the current bounds make a literal neither true nor false.

    [x >= v] is unassigned when min < v <= max, and [x <= v] is unassigned when min <= v < max.

    :param state: all the backtrackable state, whose head holds the bounds of the domains
    :type state: NDArray
    :param var_side: the variable shifted left by one, with the side of the bound in the low bit
    :type var_side: int
    :param value: the value of the literal
    :type value: int

    :return: true if the literal is unassigned
    :rtype: bool
    """
    if (var_side & 1) == DOMAIN_MIN:
        return state[var_side] < value <= state[var_side ^ 1]
    return state[var_side ^ 1] <= value < state[var_side]
