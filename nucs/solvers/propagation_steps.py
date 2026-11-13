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
The steps of one propagator call, which the propagation loops share: bc_algorithm (backtrack/bc_algorithm.py) and
lcg_algorithm (lcg/lcg_algorithm.py). One of them, update_propagators, also schedules the propagators that a decision
or a backtrack wakes.

Each step is always inlined. The two loops are compiled without the reference-counting runtime, and a callee compiled
on its own would bring the refcounts of whichever context first compiled it (see bc_algorithm).
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_NB,
    INFO_PRIORITY,
)
from nucs.solvers.buckets import STORAGE_OFFSET, buckets_add
from nucs.solvers.state import tighten_at, utrail_push, utrail_set
from nucs.statistics import (
    STATS_ALG_IDX_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
)


@njit(cache=True, inline="always")
def gather_domains(
    prop_domains: NDArray, propagator_variables: NDArray, prop_var_start: int, prop_var_end: int, state: NDArray
) -> None:
    """
    Copies the domains of a propagator's variables into its scratch buffer, as scalars.

    What this replaces is the row assignment prop_domains[var_idx] = domains[variable], and it is that assignment,
    not the row view it reads, that costs: it lowers to a generic array copy whose setup dwarfs the eight bytes it
    moves at the arity of a magic sequence's count constraints. Two scalar stores through the domains view are
    already most of the win there (90ms -> 59ms); reading the pair out of state takes it to 55ms.

    :param prop_domains: the scratch buffer, sized to the propagator's arity
    :type prop_domains: NDArray
    :param propagator_variables: the variables of every propagator, concatenated
    :type propagator_variables: NDArray
    :param prop_var_start: where the propagator's variables start in propagator_variables
    :type prop_var_start: int
    :param prop_var_end: where they end
    :type prop_var_end: int
    :param state: all the backtrackable state
    :type state: NDArray
    """
    for var_idx in range(prop_var_end - prop_var_start):
        cell_idx = propagator_variables[prop_var_start + var_idx] << 1
        prop_domains[var_idx, DOMAIN_MIN] = state[cell_idx]
        prop_domains[var_idx, DOMAIN_MAX] = state[cell_idx | 1]


@njit(cache=True, inline="always")
def utrail_state_prefix(
    state: NDArray,
    utrail_log: NDArray,
    utrail_indices: NDArray,
    mark: int,
    utrail_sz: int,
    prop_state_start: int,
    prop_state_hint: int,
) -> int:
    """
    Trails the trailed prefix of a propagator's state block before the propagator runs: its entry barrier.

    The prefix is trailed unconditionally (utrail_set no-ops if this choice point already holds a live entry for a
    cell), so the propagator itself can write its block directly with no undo trail bookkeeping of its own. The cost is
    one L1 load per trailed cell per call -- zero for a propagator with no trailed state, which today is all of
    them. The prefix bound comes out of the same offsets row as the block itself, so an empty prefix costs no load
    the slicing was not making anyway.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param utrail_sz: the number of entries currently on the undo trail
    :type utrail_sz: int
    :param prop_state_start: where the block starts in state
    :type prop_state_start: int
    :param prop_state_hint: where its untrailed hint suffix starts, the end of the prefix
    :type prop_state_hint: int

    :return: the new undo trail size
    :rtype: int
    """
    for cell in range(prop_state_start, prop_state_hint):
        utrail_sz = utrail_set(state, utrail_log, utrail_indices, mark, utrail_sz, cell, state[cell], state[cell])
    return utrail_sz


@njit(cache=True, inline="always")
def record_entailment(
    entailed: NDArray, utrail_log: NDArray, utrail_indices: NDArray, utrail_sz: int, entailed_idx: int, prop_idx: int
) -> int:
    """
    Sets a propagator's entailment flag, trailing it on the first time only.

    Entailment needs no positional guard: it is monotonic within a branch, and the test here has just established
    that the flag is still clear, so it cannot be trailed twice in a choice point.

    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param utrail_sz: the number of entries currently on the undo trail
    :type utrail_sz: int
    :param entailed_idx: where the entailment flags start in state
    :type entailed_idx: int
    :param prop_idx: the propagator
    :type prop_idx: int

    :return: the new undo trail size
    :rtype: int
    """
    if not entailed[prop_idx]:
        utrail_sz = utrail_push(utrail_log, utrail_indices, utrail_sz, entailed_idx + prop_idx, 0)
        entailed[prop_idx] = 1
    return utrail_sz


@njit(cache=True, inline="always")
def count_no_change(statistics: NDArray, algorithm_stats: int) -> None:
    """
    Counts a call that changed no domain, globally and for its algorithm.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param algorithm_stats: where the algorithm's tail of the statistics array starts
    :type algorithm_stats: int
    """
    statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1
    statistics[algorithm_stats + STATS_ALG_IDX_FILTER_NO_CHANGE_NB] += 1


# always inlined: LLVM's cost model declines to inline a function this size, but the caller is the
# per-propagator-call hot path and inlining it measurably speeds up propagator-cheap models
@njit(cache=True, inline="always")
def update_domains(
    skip_prop_idx: int,
    prop_var_start: int,
    prop_var_end: int,
    prop_domains: NDArray,
    propagator_variables: NDArray,
    state: NDArray,
    utrail_log: NDArray,
    utrail_indices: NDArray,
    mark: int,
    utrail_sz: int,
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
) -> tuple[bool, int]:
    """
    Applies a propagator's computed prop_domains and schedules the propagators triggered by the changes.

    :param skip_prop_idx: the propagator whose domains these are when it is idempotent, so that its own writes do
                          not schedule it again; -1 when it is not, so that it is rescheduled like any other
    :type skip_prop_idx: int
    :param prop_var_start: where the propagator's variables start in propagator_variables
    :type prop_var_start: int
    :param prop_var_end: where they end
    :type prop_var_end: int
    :param prop_domains: the domains computed by the propagator
    :type prop_domains: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param state: all the backtrackable state: the domain bounds, the unbound-variable count and,
                  from there on, whatever else is trailed
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param utrail_sz: the number of entries currently on the undo trail
    :type utrail_sz: int
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

    :return: true iff no domain was changed, and the new undo trail size
    :rtype: Tuple[bool, int]
    """
    no_changes = True
    for var_idx in range(prop_var_end - prop_var_start):
        variable = propagator_variables[prop_var_start + var_idx]
        # read the bounds out of state rather than out of the domains view of it: they are the same two
        # int32, but only this way can the compiler see that tighten_at is about to reload them
        cell_idx = variable << 1
        if state[cell_idx] != state[cell_idx | 1]:
            events, utrail_sz = tighten_at(
                state,
                utrail_log,
                utrail_indices,
                mark,
                utrail_sz,
                variable,
                prop_domains[var_idx, DOMAIN_MIN],
                prop_domains[var_idx, DOMAIN_MAX],
            )
            if events:
                update_propagators(
                    triggered_propagators,
                    entailed,
                    triggers,
                    triggers_offsets,
                    propagator_infos,
                    variable,
                    events,
                    skip_prop_idx,
                )
                no_changes = False
    return no_changes, utrail_sz


# always inlined: the propagation loop calls this for every variable a filtering narrows, and it is compiled
# without the reference-counting runtime, which every callee of it has to be inlined into (see bc_algorithm)
@njit(cache=True, inline="always")
def update_domains_of_copies(
    skip_prop_idx: int,
    prop_var_start: int,
    prop_var_end: int,
    prop_domains: NDArray,
    propagator_variables: NDArray,
    state: NDArray,
    utrail_log: NDArray,
    utrail_indices: NDArray,
    mark: int,
    utrail_sz: int,
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
) -> tuple[bool, bool, int]:
    """
    Applies the computed prop_domains of a propagator that has a variable more than once, and schedules the
    propagators triggered by the changes.

    The propagator sees each copy of a variable as a variable of its own. It narrows each copy soundly, so the
    variable is in the intersection of the copies, but it can narrow the copies to different bounds, even to bounds
    with an empty intersection, and still not fail. update_domains would write the copies one after the other: a later
    copy would undo the narrowing of an earlier one, and a copy that fixes the variable would hide the later copies.
    Thus this write-back:

    - intersects each copy with the current bounds, which already hold the writes of the earlier copies, and fails when
      the intersection is empty;
    - does not skip a fixed variable, because a later copy must still agree with it;
    - schedules the propagator again when a copy that it wrote is wider than the bounds that the write-back leaves,
      even when it is idempotent: its next call reads the intersection on all the copies, and at the fixpoint, it has
      accepted a box in which the copies are equal. When each copy is equal to these bounds, the propagator has
      already seen them, and an idempotent propagator is not scheduled, as with update_domains.

    The test compares each copy with the final bounds, not with the bounds at its own write: when copy 0 writes min
    1 and copy 1 then writes min 2, no write is clipped, but the propagator saw min 1 on copy 0.

    :param skip_prop_idx: the propagator whose domains these are when it is idempotent, so that its own writes
                          schedule it again only when its copies disagree; -1 when it is not, so that it is
                          rescheduled like any other
    :type skip_prop_idx: int
    :param prop_var_start: where the propagator's variables start in propagator_variables
    :type prop_var_start: int
    :param prop_var_end: where they end
    :type prop_var_end: int
    :param prop_domains: the domains computed by the propagator
    :type prop_domains: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param state: all the backtrackable state: the domain bounds, the unbound-variable count and,
                  from there on, whatever else is trailed
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param utrail_sz: the number of entries currently on the undo trail
    :type utrail_sz: int
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

    :return: false iff the copies of a variable have an empty intersection, true iff no domain was changed, and the
             new undo trail size
    :rtype: Tuple[bool, bool, int]
    """
    no_changes = True
    for var_idx in range(prop_var_end - prop_var_start):
        variable = propagator_variables[prop_var_start + var_idx]
        cell_idx = variable << 1
        new_min = max(state[cell_idx], prop_domains[var_idx, DOMAIN_MIN])
        new_max = min(state[cell_idx | 1], prop_domains[var_idx, DOMAIN_MAX])
        if new_min > new_max:
            return False, no_changes, utrail_sz
        events, utrail_sz = tighten_at(state, utrail_log, utrail_indices, mark, utrail_sz, variable, new_min, new_max)
        if events:
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                propagator_infos,
                variable,
                events,
                skip_prop_idx,
            )
            no_changes = False
    if skip_prop_idx >= 0 and not entailed[skip_prop_idx]:
        for var_idx in range(prop_var_end - prop_var_start):
            cell_idx = propagator_variables[prop_var_start + var_idx] << 1
            if (
                state[cell_idx] != prop_domains[var_idx, DOMAIN_MIN]
                or state[cell_idx | 1] != prop_domains[var_idx, DOMAIN_MAX]
            ):
                buckets_add(
                    triggered_propagators,
                    propagator_infos[skip_prop_idx, INFO_PRIORITY],
                    skip_prop_idx,
                    STORAGE_OFFSET + len(propagator_infos),
                )
                break
    return True, no_changes, utrail_sz


@njit(cache=True, inline="always")
def update_propagators(
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
    variable: int,
    events: int,
    skip_prop_idx: int,
) -> None:
    """
    Schedules the propagators that the events on a variable trigger.

    A propagator that is entailed, or already in the queue, is not scheduled. Neither is skip_prop_idx: the
    propagation loop passes the idempotent propagator whose filtering raised the events, since running it again
    on its own output would change nothing. Every other caller passes -1.

    :param triggered_propagators: the propagation queue
    :type triggered_propagators: NDArray
    :param entailed: whether each propagator is entailed
    :type entailed: NDArray
    :param triggers: the propagators to wake, grouped by (variable, event mask)
    :type triggers: NDArray
    :param triggers_offsets: the offsets of each (variable, event mask) slice of triggers
    :type triggers_offsets: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray
    :param variable: the variable whose domain changed
    :type variable: int
    :param events: the events of the change, a mask of EVENT_MASK_*
    :type events: int
    :param skip_prop_idx: a propagator not to schedule, -1 for none
    :type skip_prop_idx: int
    """
    offset = (variable << EVENT_NB) | events
    membership_offset = STORAGE_OFFSET + len(propagator_infos)
    for prop_idx in triggers[triggers_offsets[offset] : triggers_offsets[offset + 1]]:
        if not (triggered_propagators[membership_offset + prop_idx] or prop_idx == skip_prop_idx or entailed[prop_idx]):
            buckets_add(triggered_propagators, propagator_infos[prop_idx, INFO_PRIORITY], prop_idx, membership_offset)
