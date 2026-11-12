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
The explanations of the rows of the implication trail, and their debug check (design/proposals/lcg.md, 4.3 and 7.3).

To explain a row that a propagator wrote, the solver rebuilds the bounds that the propagator call read: the bounds at
the REASON_POS of the row. It gets them from the chains of the implication trail (bound_at), and it does not keep a
copy of the domains for each call. Then it calls the explain function of the propagator
(nucs.propagators.explanations) on these bounds. The explanation of a failure uses the bounds when the propagator
failed, which are the current bounds.

The explanation of a propagator is about its variables, as indices. explanation_literals changes it into literals
(var_side, value) and removes the literals that are true at the root: the conflict analysis never needs them.

A propagator can have a variable twice: in magic_sequence, count_eq has x[i] in its list and as its count. The two
copies have two indices, and the propagator can narrow one copy and not the other. The engine writes the copies into
one variable, so the row does not tell which copy the call narrowed. Thus the row also records the index of that copy
(ITRAIL_REASON_VAR_IDX), and explain gets it: the reason of [x1 >= 1] in count_eq is not the same for x1 as an element
of the list and for x1 as the count.

A wrong explanation gives a wrong learned clause, and a wrong learned clause removes solutions with no sign. Thus the
debug mode of LcgSolver checks each explanation (check_explanations): its literals must be true when the propagator
read the bounds, and they must be sufficient for the deduction.
"""

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.problems.problem import OFFSETS_PARAM, OFFSETS_STATE, OFFSETS_VARIABLE
from nucs.propagators.explanations import EXPLAIN_SIDE, EXPLAIN_VALUE, EXPLAIN_VAR_IDX, EXPLAIN_WIDTH
from nucs.propagators.propagators import call_compute_domains, call_explain
from nucs.solvers.lcg.implication_trail import (
    ITRAIL_PREV,
    ITRAIL_REASON_DATA,
    ITRAIL_REASON_KIND,
    ITRAIL_REASON_POS,
    ITRAIL_REASON_VAR_IDX,
    ITRAIL_VALUE,
    ITRAIL_VAR_SIDE,
    REASON_PROPAGATOR,
)
from nucs.solvers.lcg.literals import LIT_VALUE, LIT_VAR_SIDE, LIT_WIDTH, lit_is_true

# The verdicts of the check of an explanation, from the best to the worst. The first two say that the explanation is
# correct.
EXPLANATION_CONFIRMED = 0  # the propagation of the box of the explanation sets the literal, or fails
EXPLANATION_ENUMERATED = (
    1  # the propagation does not, but no tuple of the box satisfies the constraint and not the literal
)
EXPLANATION_SUSPICIOUS = 2  # the propagation does not, and the box has too many tuples to enumerate
EXPLANATION_WRONG = 3  # a tuple of the box satisfies the constraint and not the literal
EXPLANATION_NOT_TRUE = 4  # a literal of the explanation was not true when the propagator read the bounds
EXPLANATION_OTHER_VARIABLE = 5  # the index of the row (ITRAIL_REASON_VAR_IDX) does not name the variable of the row
EXPLANATION_NO_PROPAGATOR = 6  # the propagator of the row (ITRAIL_REASON_DATA) does not exist
EXPLANATION_VERDICTS = [
    "confirmed",
    "enumerated",
    "suspicious",
    "wrong",
    "not true",
    "on another variable",
    "on no propagator",
]

# The largest box that the check enumerates.
EXPLANATION_ENUMERATION_LIMIT = 1 << 16

# What check_explanations returns as its row when each explanation is correct.
EXPLANATIONS_VALID = -1


@njit(cache=True, inline="always")
def bound_at(itrail: NDArray, itrail_last: NDArray, initial_domains: NDArray, var_side: int, pos: int) -> int:
    """
    Returns the value that a bound had when the implication trail had pos rows.

    It walks the chain of the bound down from its last row to the first row below pos. Its cost is the number of rows
    of the bound from pos to the top.

    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param var_side: the bound
    :type var_side: int
    :param pos: a number of rows of the implication trail, at most the number of rows in use
    :type pos: int

    :return: the value of the bound
    :rtype: int
    """
    row = itrail_last[var_side]
    while row >= pos:
        row = itrail[row, ITRAIL_PREV]
    if row < 0:
        return initial_domains[var_side >> 1, var_side & 1]
    return itrail[row, ITRAIL_VALUE]


@njit(cache=True, inline="always")
def prop_variable_range(offsets: NDArray, prop_idx: int) -> tuple[int, int]:
    """
    Returns where the variables of a propagator start in propagator_variables, and how many they are.

    Both are signed. offsets is unsigned, and Numba unifies an unsigned index and -1 (no variable) into a float.
    int() keeps the type that it gets, and tells mypy that the result is an int.

    :param offsets: the CSR offsets of each propagator's slices
    :type offsets: NDArray
    :param prop_idx: the propagator
    :type prop_idx: int

    :return: the first index of the variables of the propagator in propagator_variables, and their number
    :rtype: Tuple[int, int]
    """
    prop_var_start = int(np.int64(offsets[prop_idx, OFFSETS_VARIABLE]))
    return prop_var_start, int(np.int64(offsets[prop_idx + 1, OFFSETS_VARIABLE])) - prop_var_start


@njit(cache=True)
def explain_propagator(
    explain_addrs: NDArray,
    algorithms: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_last: NDArray,
    prop_idx: int,
    pos: int,
    var_idx: int,
    side: int,
    value: int,
    prop_domains: NDArray,
    explanation: NDArray,
) -> int:
    """
    Explains a literal that a propagator call set, or the failure of a propagator call.

    It rebuilds the bounds of the variables of the propagator when the implication trail had pos rows, and calls
    the explain function of the propagator on them.

    Nothing here checks prop_idx and var_idx: the caller takes them from a REASON_PROPAGATOR row, or from
    failed_propagator. With a wrong one, the explain function reads out of its arrays, and under the JIT nothing
    reports it. The debug mode checks both (check_explanations).

    :param explain_addrs: the compiled explain address of each algorithm
    :type explain_addrs: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param offsets: the CSR offsets of each propagator's slice of propagator_variables and propagator_parameters
    :type offsets: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param prop_idx: the propagator
    :type prop_idx: int
    :param pos: the number of rows of the implication trail when the call read the bounds: the REASON_POS of a row
                that the call wrote, or the number of rows in use for a failure
    :type pos: int
    :param var_idx: the variable of the literal, as the index of the copy that the call wrote among the variables of
                    the propagator (ITRAIL_REASON_VAR_IDX), -1 for a failure
    :type var_idx: int
    :param side: DOMAIN_MIN for [x >= value], DOMAIN_MAX for [x <= value]
    :type side: int
    :param value: the value of the literal
    :type value: int
    :param prop_domains: a scratch buffer of at least as many rows as the propagator has variables; it gets the
                         bounds that the call read
    :type prop_domains: NDArray
    :param explanation: a buffer of at least two rows for each variable of the propagator; it gets the explanation,
                        one row [EXPLAIN_VAR_IDX, EXPLAIN_SIDE, EXPLAIN_VALUE] for each literal
    :type explanation: NDArray

    :return: the number of rows of the explanation
    :rtype: int
    """
    prop_var_start, arity = prop_variable_range(offsets, prop_idx)
    domains = prop_domains[:arity]
    for idx in range(arity):
        variable_side = propagator_variables[prop_var_start + idx] << 1
        domains[idx, DOMAIN_MIN] = bound_at(itrail, itrail_last, initial_domains, variable_side, pos)
        domains[idx, DOMAIN_MAX] = bound_at(itrail, itrail_last, initial_domains, variable_side | 1, pos)
    return call_explain(
        explain_addrs,
        algorithms[prop_idx],
        domains,
        propagator_parameters[offsets[prop_idx, OFFSETS_PARAM] : offsets[prop_idx + 1, OFFSETS_PARAM]],
        var_idx,
        side,
        value,
        explanation[: arity << 1],
    )


@njit(cache=True)
def explanation_literals(
    explanation: NDArray,
    row_nb: int,
    propagator_variables: NDArray,
    prop_var_start: int,
    initial_domains: NDArray,
    literals: NDArray,
) -> int:
    """
    Changes the rows of an explanation into literals, and removes the literals that are true at the root.

    :param explanation: the explanation, one row [EXPLAIN_VAR_IDX, EXPLAIN_SIDE, EXPLAIN_VALUE] for each literal
    :type explanation: NDArray
    :param row_nb: the number of rows of the explanation
    :type row_nb: int
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param prop_var_start: where the variables of the propagator start in propagator_variables
    :type prop_var_start: int
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param literals: a buffer of at least row_nb rows; it gets the literals, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type literals: NDArray

    :return: the number of literals
    :rtype: int
    """
    lit_nb = 0
    for row in range(row_nb):
        variable = propagator_variables[prop_var_start + explanation[row, EXPLAIN_VAR_IDX]]
        side = explanation[row, EXPLAIN_SIDE]
        value = explanation[row, EXPLAIN_VALUE]
        root_value = initial_domains[variable, side]
        if value <= root_value if side == DOMAIN_MIN else value >= root_value:
            continue
        literals[lit_nb, LIT_VAR_SIDE] = (variable << 1) | side
        literals[lit_nb, LIT_VALUE] = value
        lit_nb += 1
    return lit_nb


@njit(cache=True)
def check_explanation(
    compute_domains_addrs: NDArray,
    algorithm: int,
    parameters: NDArray,
    state_sz: int,
    first_copy: NDArray,
    box: NDArray,
    var_idx: int,
    side: int,
    value: int,
) -> int:
    """
    Checks that an explanation is sufficient for a literal, or for a failure: the debug check of
    design/proposals/lcg.md, 7.3.

    The box is the box of the explanation: the root bounds of the variables of the propagator, cut by the literals
    of the explanation. The check calls the propagator on the box until nothing changes, with a new state block at
    zero: one call is not sufficient when the propagator is not idempotent, or when the box is larger than the
    bounds of the call. If the propagator fails, or sets the literal, the explanation is correct. If it does not,
    an exact explanation can still be correct, because it can use reasoning that the propagator does not do. Then the
    check calls the propagator on each tuple of the box, and a tuple that does not fail must satisfy the literal (for
    a failure, no tuple must pass).

    A propagator can have a variable more than once. After each call, the check gives to all the copies of a variable
    the intersection of their bounds (propagate_to_fixpoint), and it enumerates only the tuples in which the copies
    are equal. The propagator narrows each copy soundly, so the variable is in the intersection. The write-back of the
    engine does not intersect: update_domains writes the copies one after the other with tighten_at, and skips a
    variable that an earlier copy fixed (see design/proposals/lcg.md, 7.0, step 7).

    :param compute_domains_addrs: the compiled compute_domains address of each algorithm
    :type compute_domains_addrs: NDArray
    :param algorithm: the algorithm of the propagator
    :type algorithm: int
    :param parameters: the parameters of the propagator
    :type parameters: NDArray
    :param state_sz: the size of the state block of the propagator
    :type state_sz: int
    :param first_copy: for each variable of the propagator, the first index of the same variable (see first_copies)
    :type first_copy: NDArray
    :param box: the box of the explanation, one row of bounds for each variable of the propagator, with the same
                bounds for the copies of a variable; the propagation narrows it
    :type box: NDArray
    :param var_idx: the variable of the literal, as the index of one of its copies among the variables of the
                    propagator, -1 for a failure
    :type var_idx: int
    :param side: DOMAIN_MIN for [x >= value], DOMAIN_MAX for [x <= value]
    :type side: int
    :param value: the value of the literal
    :type value: int

    :return: the verdict, one of EXPLANATION_CONFIRMED, EXPLANATION_ENUMERATED, EXPLANATION_SUSPICIOUS and
             EXPLANATION_WRONG
    :rtype: int
    """
    prop_state = np.zeros(state_sz, dtype=np.int32)
    previous = np.empty_like(box)
    status = propagate_to_fixpoint(compute_domains_addrs, algorithm, first_copy, box, parameters, prop_state, previous)
    if status == PROP_INCONSISTENCY:
        return EXPLANATION_CONFIRMED
    # var_side is the flat index of the bound in the box, as in state: a box is C-contiguous with two bounds a row
    var_side = (var_idx << 1) | side
    if var_idx >= 0 and lit_is_true(box.reshape(-1), var_side, value):
        return EXPLANATION_CONFIRMED
    # the propagation removed no solution from the box, so the tuples of the narrowed box are sufficient; a tuple
    # gives a value to the first copy of each variable only. The count is on 64 bits: without the JIT, the int32
    # bounds would give an int32 product, which wraps silently
    tuple_nb = 1
    for idx in range(len(box)):
        if first_copy[idx] == idx:
            tuple_nb *= int(np.int64(box[idx, DOMAIN_MAX])) - int(np.int64(box[idx, DOMAIN_MIN])) + 1
            if tuple_nb > EXPLANATION_ENUMERATION_LIMIT:
                return EXPLANATION_SUSPICIOUS
    point = np.empty(len(box), dtype=np.int32)
    point[:] = box[:, DOMAIN_MIN]
    singleton = np.empty_like(box)
    for _ in range(tuple_nb):
        for idx in range(len(box)):
            singleton[idx, DOMAIN_MIN] = singleton[idx, DOMAIN_MAX] = point[first_copy[idx]]
        prop_state[:] = 0
        status = propagate_to_fixpoint(
            compute_domains_addrs, algorithm, first_copy, singleton, parameters, prop_state, previous
        )
        if status != PROP_INCONSISTENCY and (var_idx < 0 or not lit_is_true(singleton.reshape(-1), var_side, value)):
            return EXPLANATION_WRONG
        # the next tuple, in the order of an odometer on the first copies
        idx = 0
        while idx < len(box) and (first_copy[idx] != idx or point[idx] == box[idx, DOMAIN_MAX]):
            if first_copy[idx] == idx:
                point[idx] = box[idx, DOMAIN_MIN]
            idx += 1
        if idx < len(box):
            point[idx] += 1
    return EXPLANATION_ENUMERATED


@njit(cache=True)
def first_copies(variables: NDArray) -> NDArray:
    """
    Returns, for each variable of a propagator, the first index of the same variable among the variables of the
    propagator.

    :param variables: the variables of the propagator
    :type variables: NDArray

    :return: the first index of each variable
    :rtype: NDArray
    """
    first_copy = np.empty(len(variables), dtype=np.int64)
    for idx in range(len(variables)):
        first_copy[idx] = variable_index(variables, 0, idx + 1, variables[idx])
    return first_copy


@njit(cache=True)
def check_explanations(
    compute_domains_addrs: NDArray,
    explain_addrs: NDArray,
    algorithms: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    first_row: int,
    failed_propagator: int,
    prop_domains: NDArray,
) -> tuple[int, int]:
    """
    Checks the explanations of the rows that the propagators wrote from a row up, and of a failure: the debug mode
    of LcgSolver.

    For each row with REASON_PROPAGATOR, it checks two literals: the literal of the row, and the weakest literal
    that the row makes true, because the conflict analysis can ask for a literal weaker than the row. For each
    literal, the literals of the explanation must be true at the REASON_POS of the row (EXPLANATION_NOT_TRUE if
    not), and they must be sufficient for the literal (check_explanation). Before that, the row must name a
    propagator (EXPLANATION_NO_PROPAGATOR if not), and its index must name the variable of the row
    (EXPLANATION_OTHER_VARIABLE if not).

    :param compute_domains_addrs: the compiled compute_domains address of each algorithm
    :type compute_domains_addrs: NDArray
    :param explain_addrs: the compiled explain address of each algorithm
    :type explain_addrs: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param offsets: the CSR offsets of each propagator's slices
    :type offsets: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param first_row: the first row to check
    :type first_row: int
    :param failed_propagator: the propagator that failed with the current bounds, -1 when there is none
    :type failed_propagator: int
    :param prop_domains: a scratch buffer with one row for each variable of the largest propagator, such as the
                         domain_buffer of the solver
    :type prop_domains: NDArray

    :return: EXPLANATIONS_VALID and EXPLANATION_CONFIRMED when each explanation is correct; else the first row whose
             explanation is not proven correct, or the number of rows in use for the failure, and the verdict
    :rtype: Tuple[int, int]
    """
    max_arity = len(prop_domains)
    explanation = np.empty((max_arity << 1, EXPLAIN_WIDTH), dtype=np.int32)
    literals = np.empty((max_arity << 1, LIT_WIDTH), dtype=np.int32)
    box = np.empty((max_arity, 2), dtype=np.int32)
    for row in range(first_row, itrail_top[0]):
        if itrail[row, ITRAIL_REASON_KIND] != REASON_PROPAGATOR:
            continue
        var_side = itrail[row, ITRAIL_VAR_SIDE]
        value = itrail[row, ITRAIL_VALUE]
        prop_idx = itrail[row, ITRAIL_REASON_DATA]
        if not 0 <= prop_idx < len(algorithms):
            return row, EXPLANATION_NO_PROPAGATOR
        # the bound before the row: [x >= before + 1] or [x <= before - 1] is the weakest literal that the row sets
        before = bound_at(itrail, itrail_last, initial_domains, var_side, row)
        weakest = before + 1 if (var_side & 1) == DOMAIN_MIN else before - 1
        for literal in range(1 if weakest == value else 2):
            lit_value = value if literal == 0 else weakest
            verdict = check_propagator_explanation(
                compute_domains_addrs,
                explain_addrs,
                algorithms,
                offsets,
                propagator_variables,
                propagator_parameters,
                initial_domains,
                itrail,
                itrail_last,
                prop_idx,
                itrail[row, ITRAIL_REASON_POS],
                itrail[row, ITRAIL_REASON_VAR_IDX],
                var_side,
                lit_value,
                prop_domains,
                explanation,
                literals,
                box,
            )
            if verdict > EXPLANATION_ENUMERATED:
                return row, verdict
    if failed_propagator >= 0:
        verdict = check_propagator_explanation(
            compute_domains_addrs,
            explain_addrs,
            algorithms,
            offsets,
            propagator_variables,
            propagator_parameters,
            initial_domains,
            itrail,
            itrail_last,
            failed_propagator,
            itrail_top[0],
            -1,
            -1,
            0,
            prop_domains,
            explanation,
            literals,
            box,
        )
        if verdict > EXPLANATION_ENUMERATED:
            return itrail_top[0], verdict
    return EXPLANATIONS_VALID, EXPLANATION_CONFIRMED


@njit(cache=True)
def check_propagator_explanation(
    compute_domains_addrs: NDArray,
    explain_addrs: NDArray,
    algorithms: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_last: NDArray,
    prop_idx: int,
    pos: int,
    var_idx: int,
    var_side: int,
    value: int,
    prop_domains: NDArray,
    explanation: NDArray,
    literals: NDArray,
    box: NDArray,
) -> int:
    """
    Explains a literal that a propagator call set, or its failure, and checks the explanation.

    :param compute_domains_addrs: the compiled compute_domains address of each algorithm
    :type compute_domains_addrs: NDArray
    :param explain_addrs: the compiled explain address of each algorithm
    :type explain_addrs: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param offsets: the CSR offsets of each propagator's slices
    :type offsets: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param prop_idx: the propagator
    :type prop_idx: int
    :param pos: the number of rows of the implication trail when the call read the bounds
    :type pos: int
    :param var_idx: the variable of the literal, as the index of the copy that the call wrote among the variables of
                    the propagator, -1 for a failure
    :type var_idx: int
    :param var_side: the bound of the literal, -1 for a failure
    :type var_side: int
    :param value: the value of the literal
    :type value: int
    :param prop_domains: a scratch buffer of max arity rows
    :type prop_domains: NDArray
    :param explanation: a scratch buffer of twice max arity rows
    :type explanation: NDArray
    :param literals: a scratch buffer of twice max arity rows
    :type literals: NDArray
    :param box: a scratch buffer of max arity rows
    :type box: NDArray

    :return: the verdict, one of EXPLANATION_*
    :rtype: int
    """
    prop_var_start, arity = prop_variable_range(offsets, prop_idx)
    side = var_side & 1
    # the index must name the variable before explain reads it
    if var_side >= 0 and (not 0 <= var_idx < arity or propagator_variables[prop_var_start + var_idx] != var_side >> 1):
        return EXPLANATION_OTHER_VARIABLE
    row_nb = explain_propagator(
        explain_addrs,
        algorithms,
        offsets,
        propagator_variables,
        propagator_parameters,
        initial_domains,
        itrail,
        itrail_last,
        prop_idx,
        pos,
        var_idx,
        side,
        value,
        prop_domains,
        explanation,
    )
    # explain_propagator left the bounds at pos in prop_domains: each row of the explanation must be true there
    domains_at_pos = prop_domains[:arity].reshape(-1)
    for row in range(row_nb):
        row_var_idx = explanation[row, EXPLAIN_VAR_IDX]
        if not 0 <= row_var_idx < arity or not lit_is_true(
            domains_at_pos, (row_var_idx << 1) | explanation[row, EXPLAIN_SIDE], explanation[row, EXPLAIN_VALUE]
        ):
            return EXPLANATION_NOT_TRUE
    lit_nb = explanation_literals(explanation, row_nb, propagator_variables, prop_var_start, initial_domains, literals)
    # the box of the explanation: the root bounds, cut by the literals on each copy of their variable
    for idx in range(arity):
        variable = propagator_variables[prop_var_start + idx]
        box[idx, DOMAIN_MIN] = initial_domains[variable, DOMAIN_MIN]
        box[idx, DOMAIN_MAX] = initial_domains[variable, DOMAIN_MAX]
        for lit in range(lit_nb):
            lit_var_side = literals[lit, LIT_VAR_SIDE]
            if lit_var_side >> 1 == variable:
                if (lit_var_side & 1) == DOMAIN_MIN:
                    box[idx, DOMAIN_MIN] = max(box[idx, DOMAIN_MIN], literals[lit, LIT_VALUE])
                else:
                    box[idx, DOMAIN_MAX] = min(box[idx, DOMAIN_MAX], literals[lit, LIT_VALUE])
    return check_explanation(
        compute_domains_addrs,
        algorithms[prop_idx],
        propagator_parameters[offsets[prop_idx, OFFSETS_PARAM] : offsets[prop_idx + 1, OFFSETS_PARAM]],
        offsets[prop_idx + 1, OFFSETS_STATE] - offsets[prop_idx, OFFSETS_STATE],
        first_copies(propagator_variables[prop_var_start : prop_var_start + arity]),
        box[:arity],
        var_idx,
        side,
        value,
    )


@njit(cache=True, inline="always")
def variable_index(propagator_variables: NDArray, prop_var_start: int, arity: int, variable: int) -> int:
    """
    Returns the first index of a variable among the variables of a propagator.

    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param prop_var_start: where the variables of the propagator start in propagator_variables
    :type prop_var_start: int
    :param arity: the number of variables of the propagator
    :type arity: int
    :param variable: the variable, one of the variables of the propagator
    :type variable: int

    :return: the index of the variable, -1 when the propagator does not have it
    :rtype: int
    """
    for idx in range(arity):
        if propagator_variables[prop_var_start + idx] == variable:
            return idx
    return -1


@njit(cache=True)
def propagate_to_fixpoint(
    compute_domains_addrs: NDArray,
    algorithm: int,
    first_copy: NDArray,
    box: NDArray,
    parameters: NDArray,
    prop_state: NDArray,
    previous: NDArray,
) -> int:
    """
    Calls a propagator on a box until it fails, it is entailed, or nothing changes.

    After each call, it gives to all the copies of a variable the intersection of their bounds, and fails if the
    intersection is empty: the propagator narrowed each copy soundly, so the variable is in the intersection.

    :param compute_domains_addrs: the compiled compute_domains address of each algorithm
    :type compute_domains_addrs: NDArray
    :param algorithm: the algorithm of the propagator
    :type algorithm: int
    :param first_copy: for each variable of the propagator, the first index of the same variable (see first_copies)
    :type first_copy: NDArray
    :param box: the bounds of the variables of the propagator, narrowed in place
    :type box: NDArray
    :param parameters: the parameters of the propagator
    :type parameters: NDArray
    :param prop_state: the state block of the propagator
    :type prop_state: NDArray
    :param previous: a scratch buffer of the shape of box
    :type previous: NDArray

    :return: the status of the last call
    :rtype: int
    """
    while True:
        previous[:] = box
        status = call_compute_domains(compute_domains_addrs, algorithm, box, parameters, prop_state)
        if status == PROP_INCONSISTENCY:
            return status
        for idx in range(len(box)):
            first = first_copy[idx]
            box[first, DOMAIN_MIN] = max(box[first, DOMAIN_MIN], box[idx, DOMAIN_MIN])
            box[first, DOMAIN_MAX] = min(box[first, DOMAIN_MAX], box[idx, DOMAIN_MAX])
            if box[first, DOMAIN_MIN] > box[first, DOMAIN_MAX]:
                return PROP_INCONSISTENCY
        for idx in range(len(box)):
            box[idx, DOMAIN_MIN] = box[first_copy[idx], DOMAIN_MIN]
            box[idx, DOMAIN_MAX] = box[first_copy[idx], DOMAIN_MAX]
        if status != PROP_CONSISTENCY or np.array_equal(previous, box):
            return status
