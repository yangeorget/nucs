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
The explanations of the propagators, for lazy clause generation (design/proposals/lcg.md, 4.3).

An explain function tells why a propagator call set a literal, or failed. It gets the bounds that the call read, and
writes a set of literals that were true in these bounds and that are sufficient for the same deduction: in each state
inside the box of these literals, the constraint forces the literal, or has no solution.

The literal to explain is on the variable at index var_idx of the propagator. When the propagator has a variable more
than once, var_idx is the index of the copy that the call narrowed, so that the explanation can use the role of that
copy in the constraint.

A row of the explanation is [EXPLAIN_VAR_IDX, EXPLAIN_SIDE, EXPLAIN_VALUE]: the literal [x >= value] for DOMAIN_MIN,
or [x <= value] for DOMAIN_MAX, on the variable x at index EXPLAIN_VAR_IDX of the propagator. An explanation has at
most one row for each bound of each variable, so the caller gives an out array of 2 * n rows for n variables.

The literals that are true at the root need no row, but an explain function can write them: the solver removes them.
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN

# The columns of a row of an explanation.
EXPLAIN_VAR_IDX = 0  # the variable of the literal, as an index into the variables of the propagator
EXPLAIN_SIDE = 1  # DOMAIN_MIN for [x >= value], DOMAIN_MAX for [x <= value]
EXPLAIN_VALUE = 2  # the value of the literal
EXPLAIN_WIDTH = 3


@njit(cache=True)
def explain_naive(domains: NDArray, parameters: NDArray, var_idx: int, side: int, value: int, out: NDArray) -> int:
    """
    Explains a literal, or a failure, with the bounds of all the variables of the propagator: the fallback
    explanation of the propagators that give no explain function.

    It is correct for each sound propagator. The propagator removed no solution of its constraint from the box
    domains, so in each state inside that box, the constraint forces the same literal, or has no solution. It needs
    no monotonicity and no idempotence. It is only long: the solver removes the literals that are true at the root.

    :param domains: the bounds of the variables of the propagator when the call read them
    :type domains: NDArray
    :param parameters: the parameters of the propagator, unused here
    :type parameters: NDArray
    :param var_idx: the variable of the literal, as an index into the variables of the propagator, -1 for a failure,
                    unused here
    :type var_idx: int
    :param side: DOMAIN_MIN for [x >= value], DOMAIN_MAX for [x <= value], unused here
    :type side: int
    :param value: the value of the literal, unused here
    :type value: int
    :param out: the explanation, one row [EXPLAIN_VAR_IDX, EXPLAIN_SIDE, EXPLAIN_VALUE] for each literal
    :type out: NDArray

    :return: the number of rows written in out, two for each variable
    :rtype: int
    """
    for idx in range(len(domains)):
        row = idx << 1
        out[row, EXPLAIN_VAR_IDX] = idx
        out[row, EXPLAIN_SIDE] = DOMAIN_MIN
        out[row, EXPLAIN_VALUE] = domains[idx, DOMAIN_MIN]
        out[row | 1, EXPLAIN_VAR_IDX] = idx
        out[row | 1, EXPLAIN_SIDE] = DOMAIN_MAX
        out[row | 1, EXPLAIN_VALUE] = domains[idx, DOMAIN_MAX]
    return len(domains) << 1
