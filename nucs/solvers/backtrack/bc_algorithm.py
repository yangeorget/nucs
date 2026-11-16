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

from nucs.constants import (
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_FLAG_IDEMPOTENT,
    PROP_FLAG_REPORTS_CHANGES,
    PROP_INCONSISTENCY,
)
from nucs.problems.problem import (
    INFO_ALGORITHM,
    INFO_COPIES,
    OFFSETS_PARAM,
    OFFSETS_STATE,
    OFFSETS_STATE_HINT,
    OFFSETS_VARIABLE,
    PROBLEM_BOUND,
    PROBLEM_INCONSISTENT,
    PROBLEM_UNBOUND,
)
from nucs.propagators.propagators import call_compute_domains
from nucs.solvers.buckets import STORAGE_OFFSET, buckets_pop
from nucs.solvers.choicepoints import CHOICEPOINT_UTRAIL_MARK
from nucs.solvers.propagation_steps import (
    count_no_change,
    gather_domains,
    record_entailment,
    update_domains,
    update_domains_of_copies,
    utrail_state_prefix,
)
from nucs.solvers.state import unbound_index
from nucs.solvers.weights import weights_bump
from nucs.statistics import (
    STATS_ALG_IDX_FILTER_NB,
    STATS_ALG_WIDTH,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_MAX,
)


# compiled without Numba's reference-counting runtime: _nrt=False, a private Numba option, which is why
# tests/solvers/backtrack/test_bc_algorithm.py pins it. With the runtime on, every array an inline="always" helper takes
# is increfed on entry and decrefed on exit, and those calls survive into the machine code -- three of each
# per bound utrail_set writes, more around every slice and every queue operation, on the hottest loop in the
# solver (see ARCHITECTURE.md). The price is two rules. Nothing in this function, or inlined into it, may allocate
# an array or touch a typed list: either fails to compile with "NRT required but not enabled", so neither can
# slip in unnoticed. And every jitted function it calls must be inline="always": one compiled on its own is
# inlined by LLVM as whichever context first compiled it into the cache left it, refcounts included -- which
# does not fail, it only makes the loop's speed depend on the cache's history.
@njit(cache=True, _nrt=False)  # type: ignore[call-overload]
def bc_algorithm(
    statistics: NDArray,
    propagator_weights: NDArray,
    algorithm_flags: NDArray,
    propagator_infos: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    state: NDArray,
    domains: NDArray,
    entailed: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    domain_buffer: NDArray,
) -> int:
    """
    This is the default consistency algorithm used by the solver.

    :param statistics: a Numpy array of statistics
    :type statistics: NDArray
    :param propagator_weights: the failure weight of each propagator, followed by the increment and its growth
                               (see nucs.solvers.weights)
    :type propagator_weights: NDArray
    :param algorithm_flags: the PROP_FLAG_* properties of each algorithm, packed into one word and indexed
                            by algorithm rather than by propagator
    :type algorithm_flags: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray
    :param offsets: the CSR offsets delimiting each propagator's slice of propagator_variables,
                    propagator_parameters and its own state block, whose trailed prefix ends at
                    OFFSETS_STATE_HINT (see OFFSETS_STATE)
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
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param compute_domains_addrs: the compiled compute_domains address of each algorithm, resolved once at solver init
    :type compute_domains_addrs: NDArray
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray

    :return: a status (consistency, inconsistency or entailment) as an integer
    :rtype: int
    """
    choicepoint = choicepoint_top[0]
    # the choice point's mark is loaded once: choicepoint_top does not move during a filtering, and neither does the
    # mark of the choice point it is filtering
    mark = choicepoints[choicepoint, CHOICEPOINT_UTRAIL_MARK]
    unbound_idx = unbound_index(state)
    # the undo trail size lives in a local for the whole filtering and is published on the way out. Nothing
    # outside this function reads it before then, and keeping it out of memory takes a load and a store
    # off every bound the propagation narrows.
    utrail_sz = utrail_top[0]
    entailed_idx = len(domains) << 1  # the entailment flags follow the domain bounds in state
    statistics[STATS_IDX_ALG_BC_NB] += 1
    membership_offset = STORAGE_OFFSET + len(propagator_infos)
    while True:
        prop_idx = buckets_pop(triggered_propagators, membership_offset)
        if prop_idx == -1:
            utrail_top[0] = utrail_sz
            return PROBLEM_BOUND if state[unbound_idx] == 0 else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        algorithm = propagator_infos[prop_idx, INFO_ALGORITHM]
        # the per-algorithm tail of the statistics array: which algorithms the calls (and, below, the
        # wasted calls) belong to, which is what makes a throughput investigation targeted
        algorithm_stats = STATS_MAX + STATS_ALG_WIDTH * algorithm
        statistics[algorithm_stats + STATS_ALG_IDX_FILTER_NB] += 1
        prop_var_start = offsets[prop_idx, OFFSETS_VARIABLE]
        prop_var_end = offsets[prop_idx + 1, OFFSETS_VARIABLE]
        prop_domains = domain_buffer[: prop_var_end - prop_var_start]
        gather_domains(prop_domains, propagator_variables, prop_var_start, prop_var_end, state)
        flags = algorithm_flags[algorithm]
        reports_changes = flags & PROP_FLAG_REPORTS_CHANGES
        prop_state_start = offsets[prop_idx, OFFSETS_STATE]
        prop_state_end = offsets[prop_idx + 1, OFFSETS_STATE]
        # where the untrailed hint suffix starts: the end of the trailed prefix, and the cell a reporting
        # propagator answers in
        prop_state_hint = offsets[prop_idx, OFFSETS_STATE_HINT]
        utrail_sz = utrail_state_prefix(
            state, utrail_log, utrail_indices, mark, utrail_sz, prop_state_start, prop_state_hint
        )
        if reports_changes:
            # pre-set to "changed", so that the fast path below is taken only on a propagator that has
            # positively said it wrote nothing -- one that forgets simply gets the scan it gets today
            state[prop_state_hint] = 1
        status = call_compute_domains(
            compute_domains_addrs,
            algorithm,
            prop_domains,
            propagator_parameters[offsets[prop_idx, OFFSETS_PARAM] : offsets[prop_idx + 1, OFFSETS_PARAM]],
            state[prop_state_start:prop_state_end],
        )
        if status == PROP_INCONSISTENCY:
            statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
            weights_bump(propagator_weights, prop_idx)
            utrail_top[0] = utrail_sz
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
            utrail_sz = record_entailment(entailed, utrail_log, utrail_indices, utrail_sz, entailed_idx, prop_idx)
        if reports_changes and status == PROP_CONSISTENCY and state[prop_state_hint] == 0:
            # the propagator wrote no domain, so update_domains would scan every variable only to find
            # no event and schedule nobody: skip it outright
            count_no_change(statistics, algorithm_stats)
            continue
        if propagator_infos[prop_idx, INFO_COPIES]:
            # the propagator has a variable more than once: its copies can disagree (see update_domains_of_copies)
            consistent, no_change, utrail_sz = update_domains_of_copies(
                prop_idx if flags & PROP_FLAG_IDEMPOTENT else -1,
                prop_var_start,
                prop_var_end,
                prop_domains,
                propagator_variables,
                state,
                utrail_log,
                utrail_indices,
                mark,
                utrail_sz,
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                propagator_infos,
            )
            if not consistent:
                statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB] += 1
                weights_bump(propagator_weights, prop_idx)
                utrail_top[0] = utrail_sz
                return PROBLEM_INCONSISTENT
        else:
            no_change, utrail_sz = update_domains(
                prop_idx if flags & PROP_FLAG_IDEMPOTENT else -1,
                prop_var_start,
                prop_var_end,
                prop_domains,
                propagator_variables,
                state,
                utrail_log,
                utrail_indices,
                mark,
                utrail_sz,
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                propagator_infos,
            )
        if no_change:
            count_no_change(statistics, algorithm_stats)
