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

from nucs.buckets import STORAGE_OFFSET, buckets_add, buckets_pop
from nucs.constants import (
    EVENT_NB,
    LEVEL_TRAIL_MARK,
    MAX,
    MIN,
    PARAM,
    PROBLEM_BOUND,
    PROBLEM_INCONSISTENT,
    PROBLEM_UNBOUND,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    STATS_ALG_IDX_FILTER_NB,
    STATS_ALG_IDX_FILTER_NO_CHANGE_NB,
    STATS_ALG_WIDTH,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_MAX,
    VARIABLE,
)
from nucs.numba_helper import ComputeDomainsFunctions
from nucs.solvers.choice_points import tighten_at, unbound_index


@njit(cache=True)
def bc_algorithm(
    propagator_nb: int,
    statistics: NDArray,
    algorithms: NDArray,
    priorities: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    state: NDArray,
    domains: NDArray,
    trail: NDArray,
    trail_top: NDArray,
    pos: NDArray,
    level_stk: NDArray,
    stks_top: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    triggered_propagators: NDArray,
    compute_domains_fcts: ComputeDomainsFunctions,
    domain_buffer: NDArray,
    idempotent: NDArray,
) -> int:
    """
    This is the default consistency algorithm used by the solver.

    :param statistics: a Numpy array of statistics
    :type statistics: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
    :param offsets: the CSR offsets delimiting each propagator's slice of propagator_variables
                    and propagator_parameters
    :type offsets: NDArray
    :param propagator_variables: the variables by propagators
    :type propagator_variables: NDArray
    :param propagator_parameters: the parameters by propagators
    :type propagator_parameters: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param state: all the backtrackable state: the domain bounds, the unbound-variable count and,
                  from there on, whatever else is trailed
    :type state: NDArray
    :param domains: the current domains, a (domain_nb, 2) view of the head of state
    :type domains: NDArray
    :param trail: the undo log of (flat index, old value) pairs
    :type trail: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param pos: the index of the last trail entry per positionally guarded cell
    :type pos: NDArray
    :param level_stk: the per-level metadata
    :type level_stk: NDArray
    :param stks_top: the height of the stacks as a Numpy array
    :type stks_top: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size,
                             the following cells hold the indices of the entailed propagators in entailment order
    :type entailment_trail: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param compute_domains_fcts: the typed list of compute_domains functions, built once at solver init
    :type compute_domains_fcts: ComputeDomainsFcts
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray
    :param idempotent: whether each algorithm reaches its own fixpoint in a single call, indexed by
                       algorithm rather than by propagator
    :type idempotent: NDArray

    :return: a status (consistency, inconsistency or entailment) as an integer
    :rtype: int
    """
    top = stks_top[0]
    # the level's mark is loaded once: stks_top does not move during a filtering, and neither does the
    # mark of the level it is filtering
    mark = level_stk[top, LEVEL_TRAIL_MARK]
    unbound = unbound_index(state)
    # the trail size lives in a local for the whole filtering and is published on the way out. Nothing
    # outside this function reads it before then, and keeping it out of memory takes a load and a store
    # off every bound the propagation narrows.
    trail_size = trail_top[0]
    statistics[STATS_IDX_ALG_BC_NB] += 1
    membership_offset = STORAGE_OFFSET + propagator_nb
    while True:
        prop_idx = buckets_pop(triggered_propagators, membership_offset)
        if prop_idx == -1:
            trail_top[0] = trail_size
            return PROBLEM_BOUND if state[unbound] == 0 else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        algorithm = algorithms[prop_idx]
        # the per-algorithm tail of the statistics array: which algorithms the calls (and, below, the
        # wasted calls) belong to, which is what makes a throughput investigation targeted
        algorithm_stats = STATS_MAX + STATS_ALG_WIDTH * algorithm
        statistics[algorithm_stats + STATS_ALG_IDX_FILTER_NB] += 1
        prop_var_start = offsets[prop_idx, VARIABLE]
        prop_var_end = offsets[prop_idx + 1, VARIABLE]
        prop_arity = prop_var_end - prop_var_start
        prop_domains = domain_buffer[:prop_arity]
        for var_idx in range(prop_arity):
            prop_domains[var_idx] = domains[propagator_variables[prop_var_start + var_idx]]
        status = compute_domains_fcts[algorithm](
            prop_domains,
            propagator_parameters[offsets[prop_idx, PARAM] : offsets[prop_idx + 1, PARAM]],
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
            trail_top[0] = trail_size
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
            if entailed_propagator_depths[prop_idx] == -1:
                # entailment is monotonic within a branch: record the shallowest depth at which the
                # propagator became entailed and push it onto the trail, so a single comparison
                # (depth != -1) detects it and a backtrack above that depth can reactivate it
                entailed_propagator_depths[prop_idx] = top
                entailment_trail[0] += 1
                entailment_trail[entailment_trail[0]] = prop_idx
        no_change, trail_size = update_domains(
            prop_idx,
            prop_var_start,
            prop_var_end,
            membership_offset,
            prop_domains,
            propagator_variables,
            state,
            trail,
            pos,
            mark,
            trail_size,
            triggered_propagators,
            entailed_propagator_depths,
            triggers,
            triggers_offsets,
            priorities,
            idempotent[algorithm],
        )
        if no_change:
            statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB] += 1
            statistics[algorithm_stats + STATS_ALG_IDX_FILTER_NO_CHANGE_NB] += 1


# always inlined: LLVM's cost model declines to inline a function this size, but the caller is the
# per-propagator-call hot path and inlining it measurably speeds up propagator-cheap models
@njit(cache=True, inline="always")
def update_domains(
    prop_idx: int,
    prop_var_start: int,
    prop_var_end: int,
    membership_offset: int,
    prop_domains: NDArray,
    propagator_variables: NDArray,
    state: NDArray,
    trail: NDArray,
    pos: NDArray,
    mark: int,
    trail_size: int,
    triggered_propagators: NDArray,
    entailed_propagator_depths: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    priorities: NDArray,
    is_idempotent: bool,
) -> tuple[bool, int]:
    """
    Applies a propagator's computed prop_domains and schedules the propagators triggered by the changes.

    :param prop_domains: the domains computed by the propagator
    :type prop_domains: NDArray
    :param is_idempotent: whether this propagator reaches its own fixpoint in a single call; when it does
                          not, it is left in its own trigger scan so it is rescheduled like any other
    :type is_idempotent: bool

    :return: true iff no domain was changed, and the new trail size
    :rtype: Tuple[bool, int]
    """
    no_changes = True
    for var_idx in range(prop_var_end - prop_var_start):
        variable = propagator_variables[prop_var_start + var_idx]
        # read the bounds out of state rather than out of the domains view of it: they are the same two
        # int32, but only this way can the compiler see that tighten is about to reload them
        flat = variable << 1
        if state[flat] != state[flat | 1]:
            events, trail_size = tighten_at(
                state,
                trail,
                pos,
                mark,
                trail_size,
                variable,
                prop_domains[var_idx, MIN],
                prop_domains[var_idx, MAX],
            )
            if events:
                offset = (variable << EVENT_NB) | events
                if is_idempotent:
                    for other_prop_idx in triggers[triggers_offsets[offset] : triggers_offsets[offset + 1]]:
                        if not (
                            triggered_propagators[membership_offset + other_prop_idx]
                            or other_prop_idx == prop_idx
                            or entailed_propagator_depths[other_prop_idx] != -1
                        ):
                            buckets_add(triggered_propagators, priorities, other_prop_idx, membership_offset)
                else:
                    for other_prop_idx in triggers[triggers_offsets[offset] : triggers_offsets[offset + 1]]:
                        if not (
                            triggered_propagators[membership_offset + other_prop_idx]
                            or entailed_propagator_depths[other_prop_idx] != -1
                        ):
                            buckets_add(triggered_propagators, priorities, other_prop_idx, membership_offset)
                no_changes = False
    return no_changes, trail_size
