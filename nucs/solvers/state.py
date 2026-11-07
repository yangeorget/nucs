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

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import EVENT_MASK_GROUND, EVENT_MASK_MAX, EVENT_MASK_MIN, EVENT_MASK_NONE

# The undo trail entries that one call of tighten_at can write: a domain's two bounds and, when it grounds the
# variable, the unbound count. Keep it in step with the function: the solver sizes the undo trail from it.
TIGHTENING_UTRAIL_ENTRY_NB = 3


@njit(cache=True, inline="always")
def unbound_index(state: NDArray) -> int:
    """
    Returns the index of the unbound-variable count, which is the last cell of the state.

    It sits at the end, after the entailment flags, rather than between them and the domains, so that
    its index does not depend on the number of propagators.

    :param state: all the backtrackable state
    :type state: NDArray

    :return: the index of the count in state
    :rtype: int
    """
    return len(state) - 1


@njit(cache=True, inline="always")
def utrail_push(utrail_log: NDArray, utrail_indices: NDArray, utrail_size: int, cell_idx: int, old: int) -> int:
    """
    Records a cell's old value and stamps its position, without testing whether the entry is needed.

    This is what utrail_set records with once its barrier has decided, and the whole of what a caller
    that owns the decision itself has to do. An entailment flag only ever goes from active to entailed,
    and only where the caller has just tested that it was active, so it cannot be written twice in a
    choice point and needs no positional guard. The position is still recorded either way, so that
    utrail_undo can clear it like any other.

    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per cell
    :type utrail_indices: NDArray
    :param utrail_size: the number of entries currently on the undo trail
    :type utrail_size: int
    :param cell_idx: the index of the cell in state
    :type cell_idx: int
    :param old: the value currently in that cell
    :type old: int

    :return: the new undo trail size
    :rtype: int
    """
    utrail_log[utrail_size, 0] = cell_idx
    utrail_log[utrail_size, 1] = old
    utrail_indices[cell_idx] = utrail_size
    return utrail_size + 1


@njit(cache=True, inline="always")
def utrail_set(
    state: NDArray,
    utrail_log: NDArray,
    utrail_indices: NDArray,
    mark: int,
    utrail_size: int,
    cell_idx: int,
    old: int,
    value: int,
) -> int:
    """
    Writes a cell of the backtrackable state, recording its old value if this choice point has not already.

    The rule a write barrier has to implement is exactly: a write to cell_idx may skip the undo trail iff the undo trail
    already holds a live entry *for cell_idx* at an index >= the current choice point's mark. Redundant entries are
    always safe -- undo is LIFO, so an extra entry restores a value at least as old as the one after it --
    but a missing entry corrupts.

    "Live" has to mean live *for this cell*, and a position inside the live range is not by itself
    evidence of that: popping the undo trail and letting it regrow leaves utrail_indices[cell_idx] pointing at an index
    some other cell has since claimed. Rather than pay to re-read the entry and check whose it is -- a random
    load on exactly the path that was about to be cheap -- utrail_undo clears the position of every entry
    it pops, so a position is stale only by being -1. That moves the cost to the pop, which happens once
    per trailed write, and off the skip, which happens as often as a fixpoint re-narrows a bound.

    Stating the rule positionally, rather than approximating it with a generation counter stamped per
    cell, means there is no counter to bump -- and therefore no site that can forget to bump one. choicepoint_init
    runs at solve time on every OPTIM_RESET, and every stack mutation is covered for free.

    The test is stated as a range, but only its lower half decides anything today: utrail_push never writes
    a position that is not below the undo trail size, and utrail_undo clears what it pops, so a position is
    either stale at -1 or live and below the size. The upper half is kept against a future pop that
    forgets to clear -- it would still reject the entries above the new top, where entry < mark alone
    would silently skip a write that needed trailing. Measured on queens, all_interval, bibd and golomb,
    dropping it changes nothing.

    The undo trail size is taken and returned rather than read out of its array, so that a caller writing
    several cells in a row -- which is every caller -- keeps it in a register instead of reloading it
    across each write.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param utrail_size: the number of entries currently on the undo trail
    :type utrail_size: int
    :param cell_idx: the index of the cell in state
    :type cell_idx: int
    :param old: the value currently in that cell, which every caller has already had to read
    :type old: int
    :param value: the value to write
    :type value: int

    :return: the new undo trail size
    :rtype: int
    """
    entry = utrail_indices[cell_idx]
    if not mark <= entry < utrail_size:  # no live entry for this cell at this choice point
        utrail_size = utrail_push(utrail_log, utrail_indices, utrail_size, cell_idx, old)
    state[cell_idx] = value
    return utrail_size


@njit(cache=True)
def utrail_undo(state: NDArray, utrail_log: NDArray, utrail_indices: NDArray, utrail_top: NDArray, mark: int) -> None:
    """
    Restores every cell the undo trail recorded since a mark, and forgets where those entries were.

    One loop, no discriminator: restoring a domain bound, the unbound-variable count and an entailed
    propagator's flag are the same pair of instructions. That is what the single flat state array buys.

    Clearing the position is what lets the write barrier trust a position that is merely in range; a
    cell whose entry has been popped is left at -1, which no range test can accept. It costs one store
    per popped entry, on the path that is already touching the entry, and saves the barrier a random
    read on every write it skips.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per cell
    :type utrail_indices: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param mark: the undo trail size to restore to
    :type mark: int
    """
    idx = utrail_top[0]
    while idx > mark:
        idx -= 1
        cell_idx = utrail_log[idx, 0]
        state[cell_idx] = utrail_log[idx, 1]
        utrail_indices[cell_idx] = -1
    utrail_top[0] = mark


@njit(cache=True, inline="always")
def tighten_at(
    state: NDArray,
    utrail_log: NDArray,
    utrail_indices: NDArray,
    mark: int,
    utrail_size: int,
    variable: int,
    new_min: int,
    new_max: int,
) -> tuple[int, int]:
    """
    Writes a variable's domain and returns the events the write triggers, taking and returning the undo trail size
    rather than reading it out of memory.

    This is the only place a backtrackable domain is written. Every kind of domain write -- a propagator's
    filtering, a branching decision, the branch-and-bound clamp, a custom consistency algorithm's pruning --
    routes through here, so the groundness test, the unbound-variable count and the write barrier stay
    consistent by construction rather than by convention.

    A filtering narrows several bounds in a row, and the propagation loop calls this for each variable of
    each propagator -- the hottest loop in the solver. Threading the undo trail size through keeps it in a
    register for the whole filtering: without that, every narrowed bound loads and stores the same cell,
    and each store is a memory dependency the next iteration has to wait on. A caller off that loop loads
    the size from utrail_top before and stores it back after.

    Scheduling is deliberately left to the caller: update_propagators does it from the events this returns,
    and only the propagation loop knows which propagator not to wake on its own writes.

    The caller owns the wipeout test too: it is one comparison on values the caller is already holding,
    and the propagation loop can never wipe out (a propagator reports the inconsistency instead), so
    charging it for the check would be paying on the hot path for the cold callers.

    :param state: all the backtrackable state
    :type state: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param mark: the undo trail size when the current choice point branched
    :type mark: int
    :param utrail_size: the number of entries currently on the undo trail
    :type utrail_size: int
    :param variable: the variable
    :type variable: int
    :param new_min: the new min of the domain
    :type new_min: int
    :param new_max: the new max of the domain
    :type new_max: int

    :return: the events triggered by the write and the new undo trail size
    :rtype: Tuple[int, int]
    """
    # the barrier is per bound, not per variable: a filtering writes both DOMAIN_MIN and DOMAIN_MAX within
    # one choice point, so a guard indexed by variable would suppress the second write and never restore
    # DOMAIN_MAX
    cell_idx = variable << 1
    old_min = state[cell_idx]
    old_max = state[cell_idx | 1]
    events = EVENT_MASK_NONE
    if old_min != new_min:
        utrail_size = utrail_set(state, utrail_log, utrail_indices, mark, utrail_size, cell_idx, old_min, new_min)
        events = EVENT_MASK_MIN
    if old_max != new_max:
        utrail_size = utrail_set(state, utrail_log, utrail_indices, mark, utrail_size, cell_idx | 1, old_max, new_max)
        events |= EVENT_MASK_MAX
    if events != EVENT_MASK_NONE and old_min != old_max and new_min == new_max:
        events |= EVENT_MASK_GROUND
        unbound_idx = unbound_index(state)
        old_unbound = state[unbound_idx]
        utrail_size = utrail_set(
            state, utrail_log, utrail_indices, mark, utrail_size, unbound_idx, old_unbound, old_unbound - 1
        )
    return events, utrail_size
