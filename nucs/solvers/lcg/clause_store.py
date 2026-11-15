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
The store of the learned clauses, and their propagation with two watched literals (design/proposals/lcg.md, 1.8, 3.5
and 4.4).

A clause is a disjunction of bound literals. The store keeps the literals of all the clauses, one after the other
(clause_lits), and the first literal of each clause (clause_start), CSR style. Clause c owns the watches 2c and 2c + 1,
which watch its literals 0 and 1: the propagation keeps the two watched literals at the first two positions of the
clause.

A literal [x >= v] becomes false when the upper bound of x goes below v, so its watch is in the list of the bound
(x, MAX): watch_head holds the first watch of each bound, and each watch holds the value of its literal and the next
watch of the list. The value lets the propagation skip a watch whose literal is not false without reading the clause.

The queue of the bounds whose watches must be visited is the implication trail itself: each row is a bound that
changed, and clause_head is the first row whose watches the propagation has not visited yet. This is the propagation
head of a SAT solver. A backtrack moves the head down to the top of the implication trail.

A learned clause with one literal has no second watch, and nothing would wake it. The store keeps such a clause in
unit_lits, and the search applies it again at each choice point that it resumes, and at each return to the root.
"""

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MIN
from nucs.solvers.lcg.conflict_analysis import first_row
from nucs.solvers.lcg.implication_trail import ITRAIL_VAR_SIDE, REASON_CLAUSE, REASON_ROOT, itrail_push_events
from nucs.solvers.lcg.literals import LIT_VALUE, LIT_VAR_SIDE, LIT_WIDTH, lit_is_false, lit_is_true, lit_negate
from nucs.solvers.propagation_steps import update_propagators
from nucs.solvers.state import tighten_at

# The columns of watches: clause c owns the rows 2c and 2c + 1.
WATCH_VALUE = 0  # the value of the watched literal
WATCH_NEXT = 1  # the next watch of the same list, -1 at the end
WATCH_WIDTH = 2

# The cells of store_top.
STORE_CLAUSE_NB = 0  # the number of clauses in the store
STORE_LIT_NB = 1  # the number of literals of these clauses
STORE_UNIT_NB = 2  # the number of clauses with one literal, in unit_lits
STORE_WIDTH = 3

# What propagate_clauses returns when no clause fails; a clause that fails returns its index instead.
CLAUSES_PROPAGATED = -1  # the propagation visited the watches of all the rows
CLAUSES_ITRAIL_FULL = -2  # the implication trail has no room for the next row


@njit(cache=True, inline="always")
def link_watch(watch_head: NDArray, watches: NDArray, watch: int, var_side: int, value: int) -> None:
    """
    Adds a watch at the head of the list of the bound whose change can make its literal false.

    :param watch_head: the first watch of each bound, -1 when the list is empty
    :type watch_head: NDArray
    :param watches: the watches, in the WATCH_* columns
    :type watches: NDArray
    :param watch: the watch
    :type watch: int
    :param var_side: the bound of the watched literal
    :type var_side: int
    :param value: the value of the watched literal
    :type value: int
    """
    bound = var_side ^ 1
    watches[watch, WATCH_VALUE] = value
    watches[watch, WATCH_NEXT] = watch_head[bound]
    watch_head[bound] = watch


@njit(cache=True, inline="always")
def swap_lits(clause_lits: NDArray, lit: int, other_lit: int) -> None:
    """
    Swaps two literals of the store.

    :param clause_lits: the literals of the clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type clause_lits: NDArray
    :param lit: the row of a literal
    :type lit: int
    :param other_lit: the row of the other literal
    :type other_lit: int
    """
    for column in range(LIT_WIDTH):
        cell = clause_lits[lit, column]
        clause_lits[lit, column] = clause_lits[other_lit, column]
        clause_lits[other_lit, column] = cell


@njit(cache=True, inline="always")
def assert_literal(
    state: NDArray,
    utrail_log: NDArray,
    utrail_indices: NDArray,
    mark: int,
    utrail_sz: int,
    itrail: NDArray,
    itrail_last: NDArray,
    itrail_sz: int,
    choicepoint: int,
    var_side: int,
    value: int,
    reason_kind: int,
    reason_data: int,
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
) -> tuple[bool, int, int]:
    """
    Makes a literal true: it tightens its bound, adds the row of the change, and schedules the propagators that the
    change triggers.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param utrail_sz: the number of entries on the undo trail
    :type utrail_sz: int
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param itrail_sz: the number of rows in use
    :type itrail_sz: int
    :param choicepoint: the current choice point, choicepoint_top[0]
    :type choicepoint: int
    :param var_side: the bound of the literal
    :type var_side: int
    :param value: the value of the literal
    :type value: int
    :param reason_kind: the reason of the row, REASON_CLAUSE or REASON_ROOT
    :type reason_kind: int
    :param reason_data: the clause for REASON_CLAUSE, -1 otherwise
    :type reason_data: int
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray

    :return: false iff the literal is false, and then nothing changes; the new undo trail size; the new number of
             rows of the implication trail
    :rtype: Tuple[bool, int, int]
    """
    if lit_is_false(state, var_side, value):
        return False, utrail_sz, itrail_sz
    variable = var_side >> 1
    cell_idx = variable << 1
    new_min = state[cell_idx]
    new_max = state[cell_idx | 1]
    if (var_side & 1) == DOMAIN_MIN:
        new_min = max(new_min, value)
    else:
        new_max = min(new_max, value)
    events, utrail_sz = tighten_at(state, utrail_log, utrail_indices, mark, utrail_sz, variable, new_min, new_max)
    if events:
        itrail_sz = itrail_push_events(
            itrail, itrail_last, itrail_sz, state, variable, events, choicepoint, reason_kind, reason_data, -1, -1
        )
        update_propagators(
            triggered_propagators, entailed, triggers, triggers_offsets, propagator_infos, variable, events, -1
        )
    return True, utrail_sz, itrail_sz


@njit(cache=True, inline="always")
def propagate_clauses(
    state: NDArray,
    utrail_log: NDArray,
    utrail_indices: NDArray,
    mark: int,
    utrail_sz: int,
    itrail: NDArray,
    itrail_last: NDArray,
    itrail_sz: int,
    itrail_reserve: int,
    choicepoint: int,
    clause_head: NDArray,
    clause_lits: NDArray,
    clause_start: NDArray,
    watch_head: NDArray,
    watches: NDArray,
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
) -> tuple[int, int, int]:
    """
    Visits the watches of the bounds that changed since clause_head, and propagates the clauses (design 4.4).

    For each watch whose literal is false: if the other watched literal is true, the clause holds; else the watch moves
    to a literal of the clause that is not false; else, if the other watched literal is false, the clause fails; else
    the other watched literal must be true, and the propagation sets it, with REASON_CLAUSE.

    The propagation stops at a failed clause, or when the implication trail has no room for one more row beside the
    reserve; the head then stays at the row that it visits, and the next call visits this row again. A second visit
    of a watch changes nothing that the first one did.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param utrail_sz: the number of entries on the undo trail
    :type utrail_sz: int
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param itrail_sz: the number of rows in use
    :type itrail_sz: int
    :param itrail_reserve: the rows to keep free for the step of the search that follows the propagation
    :type itrail_reserve: int
    :param choicepoint: the current choice point, choicepoint_top[0]
    :type choicepoint: int
    :param clause_head: the first row whose watches the propagation has not visited yet, as a one-cell array
    :type clause_head: NDArray
    :param clause_lits: the literals of the clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type clause_lits: NDArray
    :param clause_start: the first literal of each clause, and the number of literals after the last clause
    :type clause_start: NDArray
    :param watch_head: the first watch of each bound, -1 when the list is empty
    :type watch_head: NDArray
    :param watches: the watches, in the WATCH_* columns
    :type watches: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray

    :return: CLAUSES_PROPAGATED, CLAUSES_ITRAIL_FULL, or the clause that failed; the new undo trail size; the new
             number of rows of the implication trail
    :rtype: Tuple[int, int, int]
    """
    row = clause_head[0]
    while row < itrail_sz:
        bound = itrail[row, ITRAIL_VAR_SIDE]
        previous = -1
        watch = watch_head[bound]
        while watch >= 0:
            next_watch = watches[watch, WATCH_NEXT]
            # the watched literal is on the other side of the bound: [x >= v] is false when x.max < v, and [x <= v] is
            # false when x.min > v
            value = watches[watch, WATCH_VALUE]
            if (state[bound] >= value) if (bound & 1) != DOMAIN_MIN else (state[bound] <= value):
                previous = watch
                watch = next_watch
                continue
            clause = watch >> 1
            slot = watch & 1
            start = clause_start[clause]
            other = start + 1 - slot
            if lit_is_true(state, clause_lits[other, LIT_VAR_SIDE], clause_lits[other, LIT_VALUE]):
                previous = watch
                watch = next_watch
                continue
            # a literal that is not false takes the place of the watched one
            replacement = start + 2
            end = clause_start[clause + 1]
            while replacement < end and lit_is_false(
                state, clause_lits[replacement, LIT_VAR_SIDE], clause_lits[replacement, LIT_VALUE]
            ):
                replacement += 1
            if replacement < end:
                swap_lits(clause_lits, start + slot, replacement)
                if previous < 0:
                    watch_head[bound] = next_watch
                else:
                    watches[previous, WATCH_NEXT] = next_watch
                link_watch(
                    watch_head,
                    watches,
                    watch,
                    clause_lits[start + slot, LIT_VAR_SIDE],
                    clause_lits[start + slot, LIT_VALUE],
                )
                watch = next_watch
                continue
            if lit_is_false(state, clause_lits[other, LIT_VAR_SIDE], clause_lits[other, LIT_VALUE]):
                clause_head[0] = row
                return clause, utrail_sz, itrail_sz
            if itrail_sz + 1 + itrail_reserve > len(itrail):
                clause_head[0] = row
                return CLAUSES_ITRAIL_FULL, utrail_sz, itrail_sz
            _, utrail_sz, itrail_sz = assert_literal(
                state,
                utrail_log,
                utrail_indices,
                mark,
                utrail_sz,
                itrail,
                itrail_last,
                itrail_sz,
                choicepoint,
                clause_lits[other, LIT_VAR_SIDE],
                clause_lits[other, LIT_VALUE],
                REASON_CLAUSE,
                clause,
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                propagator_infos,
            )
            previous = watch
            watch = next_watch
        row += 1
    clause_head[0] = row
    return CLAUSES_PROPAGATED, utrail_sz, itrail_sz


@njit(cache=True, inline="always")
def store_clause(clause_lits: NDArray, clause_start: NDArray, store_top: NDArray, learned: NDArray, lit_nb: int) -> int:
    """
    Adds a clause of at least two literals to the store, with no watch yet: attach_clause links its watches.

    The caller makes sure that the clause fits: nothing here tests the size of the store.

    :param clause_lits: the literals of the clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type clause_lits: NDArray
    :param clause_start: the first literal of each clause, and the number of literals after the last clause
    :type clause_start: NDArray
    :param store_top: the numbers of clauses, of literals and of unit clauses, in the STORE_* cells
    :type store_top: NDArray
    :param learned: the clause, one row [LIT_VAR_SIDE, LIT_VALUE] for each literal
    :type learned: NDArray
    :param lit_nb: the number of literals of the clause, at least 2
    :type lit_nb: int

    :return: the index of the clause
    :rtype: int
    """
    clause = store_top[STORE_CLAUSE_NB]
    start = store_top[STORE_LIT_NB]
    clause_lits[start : start + lit_nb] = learned[:lit_nb]
    clause_start[clause + 1] = start + lit_nb
    store_top[STORE_CLAUSE_NB] = clause + 1
    store_top[STORE_LIT_NB] = start + lit_nb
    return clause


@njit(cache=True, inline="always")
def store_unit(unit_lits: NDArray, store_top: NDArray, var_side: int, value: int) -> None:
    """
    Adds a clause of one literal to the store of the unit clauses.

    The caller makes sure that the clause fits: nothing here tests the size of the store.

    :param unit_lits: the unit clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type unit_lits: NDArray
    :param store_top: the numbers of clauses, of literals and of unit clauses, in the STORE_* cells
    :type store_top: NDArray
    :param var_side: the bound of the literal
    :type var_side: int
    :param value: the value of the literal
    :type value: int
    """
    unit = store_top[STORE_UNIT_NB]
    unit_lits[unit, LIT_VAR_SIDE] = var_side
    unit_lits[unit, LIT_VALUE] = value
    store_top[STORE_UNIT_NB] = unit + 1


@njit(cache=True)
def attach_clause(
    clause: int,
    clause_lits: NDArray,
    clause_start: NDArray,
    watch_head: NDArray,
    watches: NDArray,
    state: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    mark: int,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    choicepoint: int,
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
) -> bool:
    """
    Links the watches of a clause that store_clause added, at the choice point where the search resumes, and sets
    its literal when the clause is unit there.

    The two watched literals are the literals that are not false, or, when fewer than two are not false, the false
    literals that became false last: when the search backtracks, these become free first. When exactly one literal is
    not false, the clause sets it, with REASON_CLAUSE. When all the literals are false, the choice point violates the
    clause: nothing is linked, and the search must backtrack again before it calls this again.

    :param clause: the clause
    :type clause: int
    :param clause_lits: the literals of the clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type clause_lits: NDArray
    :param clause_start: the first literal of each clause, and the number of literals after the last clause
    :type clause_start: NDArray
    :param watch_head: the first watch of each bound, -1 when the list is empty
    :type watch_head: NDArray
    :param watches: the watches, in the WATCH_* columns
    :type watches: NDArray
    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param choicepoint: the current choice point, choicepoint_top[0]
    :type choicepoint: int
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray

    :return: false iff all the literals of the clause are false
    :rtype: bool
    """
    start = clause_start[clause]
    end = clause_start[clause + 1]
    # the rank of a literal: the row that made it false, and above all the rows for a literal that is not false
    first = second = -1
    first_rank = second_rank = -2
    free_nb = 0
    for lit in range(start, end):
        var_side = clause_lits[lit, LIT_VAR_SIDE]
        value = clause_lits[lit, LIT_VALUE]
        if lit_is_false(state, var_side, value):
            negation_var_side, negation_value = lit_negate(var_side, value)
            rank = first_row(itrail, itrail_last, negation_var_side, negation_value)
        else:
            rank = len(itrail)
            free_nb += 1
        if rank > first_rank:
            second, second_rank = first, first_rank
            first, first_rank = lit, rank
        elif rank > second_rank:
            second, second_rank = lit, rank
    if free_nb == 0:
        return False
    # the two best literals go to the first two positions; the first swap can move the second best literal
    swap_lits(clause_lits, start, first)
    swap_lits(clause_lits, start + 1, first if second == start else second)
    for slot in range(2):
        link_watch(
            watch_head,
            watches,
            (clause << 1) | slot,
            clause_lits[start + slot, LIT_VAR_SIDE],
            clause_lits[start + slot, LIT_VALUE],
        )
    if free_nb == 1:
        _, utrail_sz, itrail_sz = assert_literal(
            state,
            utrail_log,
            utrail_indices,
            mark,
            utrail_top[0],
            itrail,
            itrail_last,
            itrail_top[0],
            choicepoint,
            clause_lits[start, LIT_VAR_SIDE],
            clause_lits[start, LIT_VALUE],
            REASON_CLAUSE,
            clause,
            triggered_propagators,
            entailed,
            triggers,
            triggers_offsets,
            propagator_infos,
        )
        utrail_top[0] = utrail_sz
        itrail_top[0] = itrail_sz
    return True


@njit(cache=True)
def apply_units(
    state: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    mark: int,
    unit_lits: NDArray,
    unit_nb: int,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    choicepoint: int,
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
) -> bool:
    """
    Applies the unit clauses at the choice point where the search resumes, or at the root, with REASON_ROOT: they
    hold for the rest of the search.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched, 0 at the root
    :type mark: int
    :param unit_lits: the unit clauses, one row [LIT_VAR_SIDE, LIT_VALUE] for each
    :type unit_lits: NDArray
    :param unit_nb: the number of unit clauses
    :type unit_nb: int
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param choicepoint: the current choice point, choicepoint_top[0]
    :type choicepoint: int
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray

    :return: false iff a unit clause is false at the choice point, which then holds no solution
    :rtype: bool
    """
    utrail_sz = utrail_top[0]
    itrail_sz = itrail_top[0]
    for unit in range(unit_nb):
        holds, utrail_sz, itrail_sz = assert_literal(
            state,
            utrail_log,
            utrail_indices,
            mark,
            utrail_sz,
            itrail,
            itrail_last,
            itrail_sz,
            choicepoint,
            unit_lits[unit, LIT_VAR_SIDE],
            unit_lits[unit, LIT_VALUE],
            REASON_ROOT,
            -1,
            triggered_propagators,
            entailed,
            triggers,
            triggers_offsets,
            propagator_infos,
        )
        if not holds:
            utrail_top[0] = utrail_sz
            itrail_top[0] = itrail_sz
            return False
    utrail_top[0] = utrail_sz
    itrail_top[0] = itrail_sz
    return True


@njit(cache=True, inline="always")
def store_has_room(
    clause_lits: NDArray,
    clause_start: NDArray,
    watches: NDArray,
    unit_lits: NDArray,
    store_top: NDArray,
    max_lit_nb: int,
) -> bool:
    """
    Returns whether the store has room for one more clause of at most max_lit_nb literals.

    :param clause_lits: the literals of the clauses
    :type clause_lits: NDArray
    :param clause_start: the first literal of each clause, and the number of literals after the last clause
    :type clause_start: NDArray
    :param watches: the watches
    :type watches: NDArray
    :param unit_lits: the unit clauses
    :type unit_lits: NDArray
    :param store_top: the numbers of clauses, of literals and of unit clauses, in the STORE_* cells
    :type store_top: NDArray
    :param max_lit_nb: the largest number of literals of a clause
    :type max_lit_nb: int

    :return: true iff each array of the store has room for one more clause
    :rtype: bool
    """
    clause_nb = store_top[STORE_CLAUSE_NB]
    return (
        store_top[STORE_LIT_NB] + max_lit_nb <= len(clause_lits)
        and clause_nb + 2 <= len(clause_start)
        and (clause_nb + 1) << 1 <= len(watches)
        and store_top[STORE_UNIT_NB] + 1 <= len(unit_lits)
    )


def allocate_clause_store(domain_nb: int) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Allocates an empty store of learned clauses.

    :param domain_nb: the number of variables of the problem
    :type domain_nb: int

    :return: clause_lits; clause_start; watch_head; watches; unit_lits; store_top; clause_head
    :rtype: Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
    """
    clause_nb = 1 << 10
    # a learned clause has at most one literal for each bound
    clause_lits = np.empty((max(1 << 14, 2 * (domain_nb << 1)), LIT_WIDTH), dtype=np.int32)
    clause_start = np.zeros(clause_nb + 1, dtype=np.int32)
    watch_head = np.full(domain_nb << 1, -1, dtype=np.int32)
    watches = np.empty((clause_nb << 1, WATCH_WIDTH), dtype=np.int32)
    unit_lits = np.empty((1 << 6, LIT_WIDTH), dtype=np.int32)
    store_top = np.zeros(STORE_WIDTH, dtype=np.int64)
    clause_head = np.zeros(1, dtype=np.int64)
    return clause_lits, clause_start, watch_head, watches, unit_lits, store_top, clause_head
