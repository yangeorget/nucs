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
The conflict analysis of lazy clause generation: the first unique implication point (design/proposals/lcg.md, 1.6 and
4.5).

The analysis starts from the nogood of a failure: the literals that the failed propagator explains. It marks them,
then walks the implication trail down. When the walk meets the row of a marked literal of the level of the conflict,
it replaces the literal with its explanation. It stops when one literal of that level is left: the first UIP. The
learned clause is the negation of the marked literals, with the negation of the UIP first: the asserting literal.

Three points are specific to bound literals and to NuCS:

- the row of a literal is the first row of the chain of its bound that makes it true, not the last change of the
  bound: [x >= 3] became true at the first row with a value of at least 3 (first_row);
- the literals on one bound merge into the strongest one: [x >= 2] and [x >= 3] together are [x >= 3]
  (mark_literal);
- a conflict can have no literal of the current level, for example when the objective bound fails at a lower level.
  The level of the conflict is then the highest level of the literals of the nogood.

The nogood of a failed clause, and the explanation of a row that a clause set (REASON_CLAUSE), are the negations of
the other literals of the clause: they need no explain function (explain_clause_and_mark).

The marks are flat arrays: for each bound, the row and the value of its marked literal (marks), and a stack of the
marked bounds (marked), which the analysis reads at the end and then clears. The walk follows the order of the
implication trail, so the analysis needs no recursion.

The levels are the choice points of the rows, and each level has one decision: from step 10 of
design/proposals/lcg.md 7.0, the search backjumps and parks no alternative, so the walk always ends at a UIP. A level
with two decisions (a hand-built trail, or the classical backtrack of steps 8 and 9) has no UIP: a decision has no
explanation, so the analysis keeps it, and the clause is valid but asserts no literal.
"""

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MIN
from nucs.problems.problem import INFO_ALGORITHM
from nucs.propagators.explanations import EXPLAIN_WIDTH
from nucs.solvers.lcg.explanations import explain_propagator, explanation_literals, prop_variable_range
from nucs.solvers.lcg.implication_trail import (
    ITRAIL_CHOICEPOINT,
    ITRAIL_PREV,
    ITRAIL_REASON_DATA,
    ITRAIL_REASON_KIND,
    ITRAIL_REASON_POS,
    ITRAIL_REASON_VAR_IDX,
    ITRAIL_VALUE,
    ITRAIL_VAR_SIDE,
    REASON_CLAUSE,
    REASON_DECISION,
    REASON_PROPAGATOR,
    REASON_ROOT,
)
from nucs.solvers.lcg.literals import LIT_VALUE, LIT_VAR_SIDE, LIT_WIDTH, lit_negate
from nucs.statistics import STATS_LCG_IDX_CLAUSE_EXPLANATION_NB, STATS_LCG_MAX

# The cells of a conflict, the start of the analysis: what failed, and the number of rows of the implication trail when
# it failed. The nogood of a conflict of kind CONFLICT_LITERAL is the literal (DATA, VALUE); the nogood of a conflict of
# kind CONFLICT_DECISIONS is the set of the decisions on the path, whose negation blocks the path.
CONFLICT_KIND = 0
CONFLICT_DATA = 1  # the propagator, the clause, or the bound of the literal
CONFLICT_VALUE = 2  # the value of the literal
CONFLICT_POS = 3
CONFLICT_WIDTH = 4

# The kinds of conflicts.
CONFLICT_NONE = -1  # no conflict waits for its analysis
CONFLICT_PROPAGATOR = 0  # a propagator failed
CONFLICT_CLAUSE = 1  # a learned clause failed
CONFLICT_LITERAL = 2  # a literal must not hold: [obj >= best] after a solution, when minimizing
CONFLICT_DECISIONS = 3  # the decisions on the path must not all hold: after a solution, or at a dead end

# The columns of marks, one row for each bound (var_side).
MARK_ROW = 0  # the row that made the marked literal true first, or MARK_NONE, or MARK_DONE
MARK_VALUE = 1  # the value of the marked literal
MARK_WIDTH = 2

# The values of MARK_ROW when the bound has no marked literal.
MARK_NONE = -1  # the bound is not on the stack of the marked bounds
MARK_DONE = -2  # the literal of the bound was replaced with its explanation; the bound is still on the stack

# The columns of clause_log_info, one row for each learned clause that the debug mode logs.
CLAUSE_LOG_START = 0  # the first row of the clause in clause_log
CLAUSE_LOG_LIT_NB = 1  # the number of literals of the clause
CLAUSE_LOG_LEVEL = 2  # the level of the conflict
CLAUSE_LOG_ASSERTION_LEVEL = 3  # the assertion level, -1 when the clause does not assert a literal
CLAUSE_LOG_SOLUTION_NB = 4  # the number of solutions that the search had found when it learned the clause
CLAUSE_LOG_WIDTH = 5

# The cells of clause_log_top.
CLAUSE_LOG_TOP_CLAUSE_NB = 0  # the number of clauses in the log
CLAUSE_LOG_TOP_LIT_NB = 1  # the number of literals in the log
CLAUSE_LOG_TOP_WIDTH = 2


@njit(cache=True, inline="always")
def set_conflict(conflict: NDArray, kind: int, data: int, value: int, pos: int) -> None:
    """
    Writes a conflict in the CONFLICT_* cells, for the analysis.

    :param conflict: the conflict
    :type conflict: NDArray
    :param kind: the kind of the conflict, one of CONFLICT_*
    :type kind: int
    :param data: the propagator, the clause, or the bound of the literal; -1 for a conflict of the decisions
    :type data: int
    :param value: the value of the literal, 0 for the other kinds
    :type value: int
    :param pos: the number of rows of the implication trail when the conflict occurred
    :type pos: int
    """
    conflict[CONFLICT_KIND] = kind
    conflict[CONFLICT_DATA] = data
    conflict[CONFLICT_VALUE] = value
    conflict[CONFLICT_POS] = pos


@njit(cache=True, inline="always")
def is_stronger(var_side: int, value: int, other_value: int) -> bool:
    """
    Returns whether a literal is strictly stronger than another literal on the same bound.

    :param var_side: the bound of the two literals
    :type var_side: int
    :param value: the value of the literal
    :type value: int
    :param other_value: the value of the other literal
    :type other_value: int

    :return: true iff the literal is strictly stronger than the other literal
    :rtype: bool
    """
    if (var_side & 1) == DOMAIN_MIN:
        return value > other_value
    return value < other_value


@njit(cache=True, inline="always")
def first_row(itrail: NDArray, itrail_last: NDArray, var_side: int, value: int) -> int:
    """
    Returns the first row of the chain of a bound that makes a literal true.

    The chain goes down from the last row of the bound, and the bound only gets tighter along the path. Thus the walk
    stops at the first row whose value does not make the literal true, and the row above it is the answer.

    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param var_side: the bound of the literal
    :type var_side: int
    :param value: the value of the literal
    :type value: int

    :return: the first row that makes the literal true, -1 when no row does: the literal is then true at the root, or
             not true at all
    :rtype: int
    """
    found = -1
    row = itrail_last[var_side]
    while row >= 0 and not is_stronger(var_side, value, itrail[row, ITRAIL_VALUE]):
        found = row
        row = itrail[row, ITRAIL_PREV]
    return found


@njit(cache=True, inline="always")
def mark_literal(
    itrail: NDArray,
    itrail_last: NDArray,
    marks: NDArray,
    marked: NDArray,
    marked_nb: int,
    var_side: int,
    value: int,
    pos: int,
    level: int,
) -> tuple[int, int, bool]:
    """
    Adds a literal of an explanation to the nogood of the analysis.

    The literal is dropped when the row that made it true first has level 0 or REASON_ROOT: it holds for the rest of
    the search. Else it merges with the marked literal of the same bound: the stronger of the two stays marked.

    The literal must be true at pos, and not at the root: explanation_literals already removes the literals that are
    true at the root. Thus a literal with no first row, or with a first row at pos or above, comes from a wrong
    explanation. To drop it would make the clause too strong, and it could remove solutions: the call refuses it.

    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param marks: for each bound, the row and the value of its marked literal, in the MARK_* columns
    :type marks: NDArray
    :param marked: the stack of the marked bounds
    :type marked: NDArray
    :param marked_nb: the number of bounds on the stack
    :type marked_nb: int
    :param var_side: the bound of the literal
    :type var_side: int
    :param value: the value of the literal
    :type value: int
    :param pos: the number of rows of the implication trail when the explained call read the bounds
    :type pos: int
    :param level: the level of the conflict, -1 when it is not known yet
    :type level: int

    :return: the new number of bounds on the stack; how much the number of marked literals of the level changes; false
             iff the literal was not true at pos, and then nothing is marked
    :rtype: Tuple[int, int, bool]
    """
    row = first_row(itrail, itrail_last, var_side, value)
    if row < 0 or row >= pos:
        return marked_nb, 0, False
    if itrail[row, ITRAIL_CHOICEPOINT] == 0 or itrail[row, ITRAIL_REASON_KIND] == REASON_ROOT:
        return marked_nb, 0, True
    level_delta = 0
    old_row = marks[var_side, MARK_ROW]
    if old_row >= 0:
        if not is_stronger(var_side, value, marks[var_side, MARK_VALUE]):
            return marked_nb, 0, True
        if itrail[old_row, ITRAIL_CHOICEPOINT] == level:
            level_delta = -1
    elif old_row == MARK_NONE:
        marked[marked_nb] = var_side
        marked_nb += 1
    marks[var_side, MARK_ROW] = row
    marks[var_side, MARK_VALUE] = value
    if itrail[row, ITRAIL_CHOICEPOINT] == level:
        level_delta += 1
    return marked_nb, level_delta, True


@njit(cache=True)
def explain_and_mark(
    lcg_statistics: NDArray,
    explain_addrs: NDArray,
    propagator_infos: NDArray,
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
    literals: NDArray,
    marks: NDArray,
    marked: NDArray,
    marked_nb: int,
    level: int,
) -> tuple[int, int, bool]:
    """
    Explains a literal that a propagator call set, or the failure of a call, and adds the literals of the explanation
    to the nogood of the analysis.

    It stops at the first literal of the explanation that was not true at pos (see mark_literal).

    :param lcg_statistics: the statistics array of the learning, see STATS_LCG_*
    :type lcg_statistics: NDArray
    :param explain_addrs: the compiled explain address of each algorithm
    :type explain_addrs: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray
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
    :param var_idx: the index of the copy that the call wrote among the variables of the propagator, -1 for a failure
    :type var_idx: int
    :param side: DOMAIN_MIN for [x >= value], DOMAIN_MAX for [x <= value]
    :type side: int
    :param value: the value of the literal
    :type value: int
    :param prop_domains: a scratch buffer with one row for each variable of the largest propagator
    :type prop_domains: NDArray
    :param explanation: a scratch buffer with two rows for each variable of the largest propagator
    :type explanation: NDArray
    :param literals: a scratch buffer with two rows for each variable of the largest propagator
    :type literals: NDArray
    :param marks: for each bound, the row and the value of its marked literal, in the MARK_* columns
    :type marks: NDArray
    :param marked: the stack of the marked bounds
    :type marked: NDArray
    :param marked_nb: the number of bounds on the stack
    :type marked_nb: int
    :param level: the level of the conflict, -1 when it is not known yet
    :type level: int

    :return: the new number of bounds on the stack; how much the number of marked literals of the level changes; false
             iff a literal of the explanation was not true at pos
    :rtype: Tuple[int, int, bool]
    """
    lcg_statistics[STATS_LCG_MAX + propagator_infos[prop_idx, INFO_ALGORITHM]] += 1
    row_nb = explain_propagator(
        explain_addrs,
        propagator_infos,
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
    prop_var_start, _ = prop_variable_range(offsets, prop_idx)
    lit_nb = explanation_literals(explanation, row_nb, propagator_variables, prop_var_start, initial_domains, literals)
    level_delta = 0
    for lit in range(lit_nb):
        marked_nb, delta, is_true = mark_literal(
            itrail,
            itrail_last,
            marks,
            marked,
            marked_nb,
            literals[lit, LIT_VAR_SIDE],
            literals[lit, LIT_VALUE],
            pos,
            level,
        )
        if not is_true:
            return marked_nb, level_delta, False
        level_delta += delta
    return marked_nb, level_delta, True


@njit(cache=True, inline="always")
def explain_clause_and_mark(
    lcg_statistics: NDArray,
    clause_lits: NDArray,
    clause_start: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_last: NDArray,
    clause: int,
    var_side: int,
    pos: int,
    marks: NDArray,
    marked: NDArray,
    marked_nb: int,
    level: int,
) -> tuple[int, int, bool]:
    """
    Explains a literal that a clause set, or the failure of a clause, and adds the literals of the explanation to the
    nogood of the analysis: the negations of the other literals of the clause.

    The negation of a literal that is false at the root is true at the root, so it is dropped, as explanation_literals
    drops such a literal. It stops at the first literal that was not true at pos (see mark_literal).

    :param lcg_statistics: the statistics array of the learning, see STATS_LCG_*
    :type lcg_statistics: NDArray
    :param clause_lits: the literals of the clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type clause_lits: NDArray
    :param clause_start: the first literal of each clause, and the number of literals after the last clause
    :type clause_start: NDArray
    :param initial_domains: the domains of the problem before any propagation, the root values of the bounds
    :type initial_domains: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param clause: the clause
    :type clause: int
    :param var_side: the bound that the clause set, -1 for a failure
    :type var_side: int
    :param pos: the row that the clause set, or the number of rows in use for a failure: the other literals were false
                before it
    :type pos: int
    :param marks: for each bound, the row and the value of its marked literal, in the MARK_* columns
    :type marks: NDArray
    :param marked: the stack of the marked bounds
    :type marked: NDArray
    :param marked_nb: the number of bounds on the stack
    :type marked_nb: int
    :param level: the level of the conflict, -1 when it is not known yet
    :type level: int

    :return: the new number of bounds on the stack; how much the number of marked literals of the level changes; false
             iff a literal of the explanation was not true at pos
    :rtype: Tuple[int, int, bool]
    """
    lcg_statistics[STATS_LCG_IDX_CLAUSE_EXPLANATION_NB] += 1
    level_delta = 0
    for lit in range(clause_start[clause], clause_start[clause + 1]):
        lit_var_side = clause_lits[lit, LIT_VAR_SIDE]
        if lit_var_side == var_side:
            continue
        negation_var_side, negation_value = lit_negate(lit_var_side, clause_lits[lit, LIT_VALUE])
        root_value = initial_domains[negation_var_side >> 1, negation_var_side & 1]
        if not is_stronger(negation_var_side, negation_value, root_value):
            continue
        marked_nb, delta, is_true = mark_literal(
            itrail, itrail_last, marks, marked, marked_nb, negation_var_side, negation_value, pos, level
        )
        if not is_true:
            return marked_nb, level_delta, False
        level_delta += delta
    return marked_nb, level_delta, True


@njit(cache=True, inline="always")
def mark_decisions(
    itrail: NDArray,
    itrail_last: NDArray,
    pos: int,
    marks: NDArray,
    marked: NDArray,
) -> tuple[int, bool]:
    """
    Marks the decisions on the path, the nogood of a conflict of kind CONFLICT_DECISIONS: the rows with
    REASON_DECISION, one for each level.

    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param pos: the number of rows of the implication trail when the conflict occurred
    :type pos: int
    :param marks: for each bound, the row and the value of its marked literal, in the MARK_* columns
    :type marks: NDArray
    :param marked: the stack of the marked bounds
    :type marked: NDArray

    :return: the number of bounds on the stack, and false iff a decision was not true at pos
    :rtype: Tuple[int, bool]
    """
    marked_nb = 0
    for row in range(pos):
        if itrail[row, ITRAIL_REASON_KIND] == REASON_DECISION:
            marked_nb, _, is_true = mark_literal(
                itrail,
                itrail_last,
                marks,
                marked,
                marked_nb,
                itrail[row, ITRAIL_VAR_SIDE],
                itrail[row, ITRAIL_VALUE],
                pos,
                -1,
            )
            if not is_true:
                return marked_nb, False
    return marked_nb, True


@njit(cache=True)
def analyze_conflict(
    lcg_statistics: NDArray,
    explain_addrs: NDArray,
    propagator_infos: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    conflict: NDArray,
    clause_lits: NDArray,
    clause_start: NDArray,
    prop_domains: NDArray,
    explanation: NDArray,
    literals: NDArray,
    marks: NDArray,
    marked: NDArray,
    learned: NDArray,
    level_stamps: NDArray,
    stamp: int,
) -> tuple[int, int, int, int]:
    """
    Analyzes a conflict, and returns the learned clause: the 1UIP analysis of design/proposals/lcg.md 4.5. The conflict
    is the failure of a propagator call or of a clause, a literal that must not hold, or the decisions of the path.

    The learned clause is the negation of the final nogood. When the analysis finds a UIP, the negation of the UIP is
    the first literal of the clause, and the assertion level is the highest level of the other literals, 0 when there
    is none. An empty clause means that no solution is left.

    When an explanation gives a literal that was not true when its call read the bounds, the explanation is wrong, and
    the analysis gives no clause: it returns -1 literals, and the row whose explanation is wrong, or the number of
    rows in use for the explanation of the failure.

    :param lcg_statistics: the statistics array of the learning, see STATS_LCG_*
    :type lcg_statistics: NDArray
    :param explain_addrs: the compiled explain address of each algorithm
    :type explain_addrs: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray
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
    :param conflict: the conflict, in the CONFLICT_* cells: for a propagator, its POS is the number of rows of the
                     implication trail when the call that failed read the bounds; for the other kinds, the number of
                     rows when the conflict occurred
    :type conflict: NDArray
    :param clause_lits: the literals of the learned clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type clause_lits: NDArray
    :param clause_start: the first literal of each learned clause
    :type clause_start: NDArray
    :param prop_domains: a scratch buffer with one row for each variable of the largest propagator
    :type prop_domains: NDArray
    :param explanation: a scratch buffer with two rows for each variable of the largest propagator
    :type explanation: NDArray
    :param literals: a scratch buffer with two rows for each variable of the largest propagator
    :type literals: NDArray
    :param marks: for each bound, the row and the value of its marked literal, in the MARK_* columns, with MARK_ROW at
                  MARK_NONE; the analysis leaves it so
    :type marks: NDArray
    :param marked: a scratch buffer with one cell for each bound: the stack of the marked bounds
    :type marked: NDArray
    :param learned: a buffer with one row for each bound; it gets the learned clause, one row [LIT_VAR_SIDE, LIT_VALUE]
                    for each literal
    :type learned: NDArray
    :param level_stamps: one cell for each level; the analysis writes stamp in the cells of the levels of an
                         asserting clause, to count them
    :type level_stamps: NDArray
    :param stamp: a value that no cell of level_stamps holds
    :type stamp: int

    :return: the number of literals of the learned clause, -1 when an explanation is wrong; the level of the conflict,
             the highest level of the literals of the nogood of the failure, 0 when there is none, or the row whose
             explanation is wrong; the assertion level, -1 when the clause does not assert a literal; the LBD of
             the clause, its number of distinct levels, 0 when it does not assert a literal
    :rtype: Tuple[int, int, int, int]
    """
    # the nogood of the failure; its level is not known before all its literals are marked
    conflict_kind = conflict[CONFLICT_KIND]
    if conflict_kind == CONFLICT_PROPAGATOR:
        marked_nb, _, is_true = explain_and_mark(
            lcg_statistics,
            explain_addrs,
            propagator_infos,
            offsets,
            propagator_variables,
            propagator_parameters,
            initial_domains,
            itrail,
            itrail_last,
            conflict[CONFLICT_DATA],
            conflict[CONFLICT_POS],
            -1,
            -1,
            0,
            prop_domains,
            explanation,
            literals,
            marks,
            marked,
            0,
            -1,
        )
    elif conflict_kind == CONFLICT_CLAUSE:
        marked_nb, _, is_true = explain_clause_and_mark(
            lcg_statistics,
            clause_lits,
            clause_start,
            initial_domains,
            itrail,
            itrail_last,
            conflict[CONFLICT_DATA],
            -1,
            conflict[CONFLICT_POS],
            marks,
            marked,
            0,
            -1,
        )
    elif conflict_kind == CONFLICT_DECISIONS:
        marked_nb, is_true = mark_decisions(itrail, itrail_last, conflict[CONFLICT_POS], marks, marked)
    else:
        # [obj >= best] after a solution: the nogood is the reason of the literal, not the literal itself, which would
        # be its own UIP and would give the bound as a unit clause, and a jump back to the root after each solution.
        # With its reason, the analysis learns why the objective was so large (design/proposals/lcg.md, 4.7).
        marked_nb = 0
        is_true = True
        var_side = conflict[CONFLICT_DATA]
        value = conflict[CONFLICT_VALUE]
        if is_stronger(var_side, value, initial_domains[var_side >> 1, var_side & 1]):
            row = first_row(itrail, itrail_last, var_side, value)
            if row < 0 or row >= conflict[CONFLICT_POS]:
                is_true = False
            elif itrail[row, ITRAIL_CHOICEPOINT] == 0 or itrail[row, ITRAIL_REASON_KIND] == REASON_ROOT:
                pass  # the literal holds for the rest of the search: no better solution is left
            elif itrail[row, ITRAIL_REASON_KIND] == REASON_PROPAGATOR:
                marked_nb, _, is_true = explain_and_mark(
                    lcg_statistics,
                    explain_addrs,
                    propagator_infos,
                    offsets,
                    propagator_variables,
                    propagator_parameters,
                    initial_domains,
                    itrail,
                    itrail_last,
                    itrail[row, ITRAIL_REASON_DATA],
                    itrail[row, ITRAIL_REASON_POS],
                    itrail[row, ITRAIL_REASON_VAR_IDX],
                    var_side & 1,
                    value,
                    prop_domains,
                    explanation,
                    literals,
                    marks,
                    marked,
                    0,
                    -1,
                )
            elif itrail[row, ITRAIL_REASON_KIND] == REASON_CLAUSE:
                marked_nb, _, is_true = explain_clause_and_mark(
                    lcg_statistics,
                    clause_lits,
                    clause_start,
                    initial_domains,
                    itrail,
                    itrail_last,
                    itrail[row, ITRAIL_REASON_DATA],
                    var_side,
                    row,
                    marks,
                    marked,
                    0,
                    -1,
                )
            else:  # a decision has no reason: it is the nogood
                marked_nb, _, is_true = mark_literal(
                    itrail, itrail_last, marks, marked, 0, var_side, value, conflict[CONFLICT_POS], -1
                )
    if not is_true:
        return clear_marks(marks, marked, marked_nb, itrail_top[0])
    if marked_nb == 0:
        return 0, 0, -1, 0
    # the level of the conflict, and its marked literals that the walk has not reached yet
    level = 0
    level_lit_nb = 0
    for idx in range(marked_nb):
        lit_level = itrail[marks[marked[idx], MARK_ROW], ITRAIL_CHOICEPOINT]
        if lit_level > level:
            level = lit_level
            level_lit_nb = 1
        elif lit_level == level:
            level_lit_nb += 1
    # the decisions of the level that stay in the nogood, because they have no explanation
    kept_nb = 0
    uip = -1
    row = itrail_top[0] - 1
    # the rows of a level are contiguous, and the rows above it have higher levels: the walk visits all the marked
    # literals of the level before any marked literal of a lower level
    while level_lit_nb > 0 and row >= 0 and itrail[row, ITRAIL_CHOICEPOINT] >= level:
        var_side = itrail[row, ITRAIL_VAR_SIDE]
        if marks[var_side, MARK_ROW] == row:
            if level_lit_nb == 1 and kept_nb == 0:
                uip = var_side
                break
            level_lit_nb -= 1
            if itrail[row, ITRAIL_REASON_KIND] == REASON_PROPAGATOR:
                value = marks[var_side, MARK_VALUE]
                marks[var_side, MARK_ROW] = MARK_DONE
                # the analysis asks for the marked literal, which can be weaker than the literal of the row
                marked_nb, level_delta, is_true = explain_and_mark(
                    lcg_statistics,
                    explain_addrs,
                    propagator_infos,
                    offsets,
                    propagator_variables,
                    propagator_parameters,
                    initial_domains,
                    itrail,
                    itrail_last,
                    itrail[row, ITRAIL_REASON_DATA],
                    itrail[row, ITRAIL_REASON_POS],
                    itrail[row, ITRAIL_REASON_VAR_IDX],
                    var_side & 1,
                    value,
                    prop_domains,
                    explanation,
                    literals,
                    marks,
                    marked,
                    marked_nb,
                    level,
                )
                if not is_true:
                    return clear_marks(marks, marked, marked_nb, row)
                level_lit_nb += level_delta
            elif itrail[row, ITRAIL_REASON_KIND] == REASON_CLAUSE:
                marks[var_side, MARK_ROW] = MARK_DONE
                marked_nb, level_delta, is_true = explain_clause_and_mark(
                    lcg_statistics,
                    clause_lits,
                    clause_start,
                    initial_domains,
                    itrail,
                    itrail_last,
                    itrail[row, ITRAIL_REASON_DATA],
                    var_side,
                    row,
                    marks,
                    marked,
                    marked_nb,
                    level,
                )
                if not is_true:
                    return clear_marks(marks, marked, marked_nb, row)
                level_lit_nb += level_delta
            else:
                kept_nb += 1
        row -= 1
    # the learned clause, with the asserting literal first when there is a UIP
    lit_nb = 0
    assertion_level = -1
    lbd = 0
    if uip >= 0:
        learned[0, LIT_VAR_SIDE], learned[0, LIT_VALUE] = lit_negate(uip, marks[uip, MARK_VALUE])
        lit_nb = 1
        assertion_level = 0
        level_stamps[level] = stamp
        lbd = 1
    for idx in range(marked_nb):
        var_side = marked[idx]
        mark_row = marks[var_side, MARK_ROW]
        if mark_row >= 0 and var_side != uip:
            learned[lit_nb, LIT_VAR_SIDE], learned[lit_nb, LIT_VALUE] = lit_negate(
                var_side, marks[var_side, MARK_VALUE]
            )
            lit_nb += 1
            if uip >= 0:
                lit_level = itrail[mark_row, ITRAIL_CHOICEPOINT]
                assertion_level = max(assertion_level, lit_level)
                if level_stamps[lit_level] != stamp:
                    level_stamps[lit_level] = stamp
                    lbd += 1
        marks[var_side, MARK_ROW] = MARK_NONE
    return lit_nb, level, assertion_level, lbd


@njit(cache=True, inline="always")
def clear_marks(marks: NDArray, marked: NDArray, marked_nb: int, row: int) -> tuple[int, int, int, int]:
    """
    Clears the marks of an analysis that found a wrong explanation, and returns its result: no clause.

    :param marks: for each bound, the row and the value of its marked literal, in the MARK_* columns
    :type marks: NDArray
    :param marked: the stack of the marked bounds
    :type marked: NDArray
    :param marked_nb: the number of bounds on the stack
    :type marked_nb: int
    :param row: the row whose explanation is wrong, or the number of rows in use for the explanation of the failure
    :type row: int

    :return: -1 literals, the row, no assertion level, and no LBD
    :rtype: Tuple[int, int, int, int]
    """
    for idx in range(marked_nb):
        marks[marked[idx], MARK_ROW] = MARK_NONE
    return -1, row, -1, 0


@njit(cache=True, inline="always")
def clause_log_add(
    clause_log: NDArray,
    clause_log_info: NDArray,
    clause_log_top: NDArray,
    learned: NDArray,
    lit_nb: int,
    level: int,
    assertion_level: int,
    solution_nb: int,
) -> None:
    """
    Adds a learned clause to the log of the debug mode.

    The caller makes sure that the clause fits: nothing here tests the size of the log.

    :param clause_log: the literals of the logged clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type clause_log: NDArray
    :param clause_log_info: one row for each logged clause, in the CLAUSE_LOG_* columns
    :type clause_log_info: NDArray
    :param clause_log_top: the number of clauses and the number of literals in the log, in the CLAUSE_LOG_TOP_* cells
    :type clause_log_top: NDArray
    :param learned: the learned clause, one row [LIT_VAR_SIDE, LIT_VALUE] for each literal
    :type learned: NDArray
    :param lit_nb: the number of literals of the learned clause
    :type lit_nb: int
    :param level: the level of the conflict
    :type level: int
    :param assertion_level: the assertion level, -1 when the clause does not assert a literal
    :type assertion_level: int
    :param solution_nb: the number of solutions that the search has found
    :type solution_nb: int
    """
    clause = clause_log_top[CLAUSE_LOG_TOP_CLAUSE_NB]
    start = clause_log_top[CLAUSE_LOG_TOP_LIT_NB]
    clause_log[start : start + lit_nb] = learned[:lit_nb]
    clause_log_info[clause, CLAUSE_LOG_START] = start
    clause_log_info[clause, CLAUSE_LOG_LIT_NB] = lit_nb
    clause_log_info[clause, CLAUSE_LOG_LEVEL] = level
    clause_log_info[clause, CLAUSE_LOG_ASSERTION_LEVEL] = assertion_level
    clause_log_info[clause, CLAUSE_LOG_SOLUTION_NB] = solution_nb
    clause_log_top[CLAUSE_LOG_TOP_CLAUSE_NB] = clause + 1
    clause_log_top[CLAUSE_LOG_TOP_LIT_NB] = start + lit_nb


def allocate_analysis(domain_nb: int, max_arity: int) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Allocates the buffers of the conflict analysis of a problem.

    :param domain_nb: the number of variables of the problem
    :type domain_nb: int
    :param max_arity: the number of variables of the largest propagator of the problem
    :type max_arity: int

    :return: the scratch buffer of the explanations; the scratch buffer of their literals; the marks, with MARK_ROW
             at MARK_NONE; the stack of the marked bounds; the learned clause
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]
    """
    explanation = np.empty((max_arity << 1, EXPLAIN_WIDTH), dtype=np.int32)
    literals = np.empty((max_arity << 1, LIT_WIDTH), dtype=np.int32)
    marks = np.full((domain_nb << 1, MARK_WIDTH), MARK_NONE, dtype=np.int32)
    marked = np.empty(domain_nb << 1, dtype=np.int32)
    learned = np.empty((domain_nb << 1, LIT_WIDTH), dtype=np.int32)
    return explanation, literals, marks, marked, learned


def allocate_clause_log(domain_nb: int, debug: bool) -> tuple[NDArray, NDArray, NDArray]:
    """
    Allocates the log of the learned clauses of the debug mode.

    :param domain_nb: the number of variables of the problem
    :type domain_nb: int
    :param debug: whether the solver runs in debug mode; without it, nothing writes the log, which gets no rows
    :type debug: bool

    :return: the literals of the logged clauses; one row for each logged clause, in the CLAUSE_LOG_* columns; the
             number of clauses and the number of literals in the log
    :rtype: Tuple[NDArray, NDArray, NDArray]
    """
    # a learned clause has at most one literal for each bound
    clause_log = np.empty((max(1 << 10, 2 * (domain_nb << 1)) if debug else 0, LIT_WIDTH), dtype=np.int32)
    clause_log_info = np.empty((1 << 8 if debug else 0, CLAUSE_LOG_WIDTH), dtype=np.int64)
    clause_log_top = np.zeros(CLAUSE_LOG_TOP_WIDTH, dtype=np.int64)
    return clause_log, clause_log_info, clause_log_top
