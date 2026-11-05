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
lcg_algorithm (lcg/lcg_algorithm.py).

Each step is always inlined. The two loops are compiled without the reference-counting runtime, and a callee compiled
on its own would bring the refcounts of whichever context first compiled it (see bc_algorithm).
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
)
from nucs.propagators.propagators import update_propagators
from nucs.solvers.state import tighten_at, trail_push, trail_set
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
def trail_state_prefix(
    state: NDArray,
    trail_log: NDArray,
    trail_indices: NDArray,
    mark: int,
    trail_size: int,
    prop_state_start: int,
    prop_state_hint: int,
) -> int:
    """
    Trails the trailed prefix of a propagator's state block before the propagator runs: its entry barrier.

    The prefix is trailed unconditionally (trail_set no-ops if this choice point already holds a live entry for a
    cell), so the propagator itself can write its block directly with no trail bookkeeping of its own. The cost is
    one L1 load per trailed cell per call -- zero for a propagator with no trailed state, which today is all of
    them. The prefix bound comes out of the same offsets row as the block itself, so an empty prefix costs no load
    the slicing was not making anyway.

    :param state: all the backtrackable state
    :type state: NDArray
    :param trail_log: the undo log of (cell index, old value) pairs
    :type trail_log: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param mark: the trail size when the current choice point branched
    :type mark: int
    :param trail_size: the number of entries currently on the trail
    :type trail_size: int
    :param prop_state_start: where the block starts in state
    :type prop_state_start: int
    :param prop_state_hint: where its untrailed hint suffix starts, the end of the prefix
    :type prop_state_hint: int

    :return: the new trail size
    :rtype: int
    """
    for cell in range(prop_state_start, prop_state_hint):
        trail_size = trail_set(state, trail_log, trail_indices, mark, trail_size, cell, state[cell], state[cell])
    return trail_size


@njit(cache=True, inline="always")
def record_entailment(
    entailed: NDArray, trail_log: NDArray, trail_indices: NDArray, trail_size: int, entailed_idx: int, prop_idx: int
) -> int:
    """
    Sets a propagator's entailment flag, trailing it on the first time only.

    Entailment needs no positional guard: it is monotonic within a branch, and the test here has just established
    that the flag is still clear, so it cannot be trailed twice in a choice point.

    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param trail_log: the undo log of (cell index, old value) pairs
    :type trail_log: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param trail_size: the number of entries currently on the trail
    :type trail_size: int
    :param entailed_idx: where the entailment flags start in state
    :type entailed_idx: int
    :param prop_idx: the propagator
    :type prop_idx: int

    :return: the new trail size
    :rtype: int
    """
    if not entailed[prop_idx]:
        trail_size = trail_push(trail_log, trail_indices, trail_size, entailed_idx + prop_idx, 0)
        entailed[prop_idx] = 1
    return trail_size


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
    trail_log: NDArray,
    trail_indices: NDArray,
    mark: int,
    trail_size: int,
    triggered_propagators: NDArray,
    entailed: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    priorities: NDArray,
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
    :param trail_log: the undo log of (cell index, old value) pairs
    :type trail_log: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param mark: the trail size when the current choice point branched
    :type mark: int
    :param trail_size: the number of entries currently on the trail
    :type trail_size: int
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray

    :return: true iff no domain was changed, and the new trail size
    :rtype: Tuple[bool, int]
    """
    no_changes = True
    for var_idx in range(prop_var_end - prop_var_start):
        variable = propagator_variables[prop_var_start + var_idx]
        # read the bounds out of state rather than out of the domains view of it: they are the same two
        # int32, but only this way can the compiler see that tighten_at is about to reload them
        cell_idx = variable << 1
        if state[cell_idx] != state[cell_idx | 1]:
            events, trail_size = tighten_at(
                state,
                trail_log,
                trail_indices,
                mark,
                trail_size,
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
                    priorities,
                    variable,
                    events,
                    skip_prop_idx,
                )
                no_changes = False
    return no_changes, trail_size
