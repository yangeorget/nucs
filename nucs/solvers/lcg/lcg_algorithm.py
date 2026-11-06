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
The propagation loop of :class:`LcgSolver`.
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_FLAG_IDEMPOTENT,
    PROP_FLAG_REPORTS_CHANGES,
    PROP_INCONSISTENCY,
)
from nucs.problems.problem import (
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
from nucs.solvers.choice_points import CHOICE_POINT_TRAIL_MARK
from nucs.solvers.lcg.implication_trail import LIT_STEP_ROW_NB, REASON_PROPAGATOR, lit_push_events
from nucs.solvers.propagation_steps import (
    count_no_change,
    gather_domains,
    record_entailment,
    trail_state_prefix,
    update_propagators,
)
from nucs.solvers.state import tighten_at, unbound_index
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

# returned by lcg_algorithm when the implication trail has no room for the next propagator call; the propagator is
# still in the queue, and the next call of lcg_algorithm continues the propagation
PROBLEM_LIT_TRAIL_FULL = 3


# compiled without Numba's reference-counting runtime, as bc_algorithm is, for the same reasons and with the same two
# rules (see bc_algorithm): nothing in it may allocate or touch a typed list, and every jitted function it calls must be
# inline="always". tests/solvers/lcg/test_lcg_algorithm.py checks both.
@njit(cache=True, _nrt=False)  # type: ignore[call-overload]
def lcg_algorithm(
    statistics: NDArray,
    propagator_weights: NDArray,
    algorithm_flags: NDArray,
    algorithms: NDArray,
    priorities: NDArray,
    offsets: NDArray,
    propagator_variables: NDArray,
    propagator_parameters: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    state: NDArray,
    domains: NDArray,
    entailed: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    choice_point_stk: NDArray,
    choice_point_top: NDArray,
    triggered_propagators: NDArray,
    compute_domains_addrs: NDArray,
    domain_buffer: NDArray,
    lit_trail: NDArray,
    lit_top: NDArray,
    lit_last: NDArray,
) -> int:
    """
    Propagates as bc_algorithm does, and records each bound change on the implication trail: the propagation loop
    of LcgSolver.

    It is a copy of bc_algorithm with two changes (design/proposals/lcg.md, 4.1):

    - before the call of a propagator, it keeps the number of rows of the implication trail as the REASON_POS of
      the call;
    - in the write-back, each moved bound adds a row with REASON_PROPAGATOR, the propagator and this REASON_POS.

    The search is the same as with bc_algorithm, and so are the counters: it counts its calls in ALG_BC_NB for this
    reason.

    Each propagator call adds at most two rows for each of its variables. Before it pops a propagator, the loop makes
    sure that these rows fit, and that the rows of the next step of the search fit too. When they do not, it returns
    PROBLEM_LIT_TRAIL_FULL and leaves the propagator in the queue: the solver grows the trail, and the next call
    continues the propagation. The two calls count as one in ALG_BC_NB.

    :param statistics: a Numpy array of statistics
    :type statistics: NDArray
    :param propagator_weights: the failure weight of each propagator, followed by the increment and its growth
                               (see nucs.solvers.weights)
    :type propagator_weights: NDArray
    :param algorithm_flags: the PROP_FLAG_* properties of each algorithm, packed into one word and indexed
                            by algorithm rather than by propagator
    :type algorithm_flags: NDArray
    :param algorithms: the algorithms indexed by propagators
    :type algorithms: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
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
    :param trail_log: the undo log of (cell index, old value) pairs
    :type trail_log: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param trail_indices: the index of the last trail entry per positionally guarded cell
    :type trail_indices: NDArray
    :param choice_point_stk: the per-choice-point metadata
    :type choice_point_stk: NDArray
    :param choice_point_top: the index of the top of the choice points as a Numpy array
    :type choice_point_top: NDArray
    :param triggered_propagators: the Numpy array of triggered propagators
    :type triggered_propagators: NDArray
    :param compute_domains_addrs: the compiled compute_domains address of each algorithm, resolved once at solver init
    :type compute_domains_addrs: NDArray
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray
    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_top: the number of rows in use, as a one-cell array
    :type lit_top: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray

    :return: PROBLEM_INCONSISTENT, PROBLEM_UNBOUND, PROBLEM_BOUND, or PROBLEM_LIT_TRAIL_FULL when the implication
             trail must grow before the propagation can continue
    :rtype: int
    """
    choice_point = choice_point_top[0]
    # the choice point's mark is loaded once: choice_point_top does not move during a filtering, and neither does the
    # mark of the choice point it is filtering
    mark = choice_point_stk[choice_point, CHOICE_POINT_TRAIL_MARK]
    unbound_idx = unbound_index(state)
    # the trail size lives in a local for the whole filtering and is published on the way out. Nothing
    # outside this function reads it before then, and keeping it out of memory takes a load and a store
    # off every bound the propagation narrows.
    trail_size = trail_top[0]
    entailed_idx = len(domains) << 1  # the entailment flags follow the domain bounds in state
    # the same for the implication trail
    lit_size = lit_top[0]
    # the rows that the next call can add, and those of the step of the search that follows the propagation
    lit_room = 2 * len(domain_buffer) + LIT_STEP_ROW_NB
    statistics[STATS_IDX_ALG_BC_NB] += 1
    membership_offset = STORAGE_OFFSET + len(priorities)
    while True:
        if lit_size + lit_room > len(lit_trail):
            statistics[STATS_IDX_ALG_BC_NB] -= 1  # the next call continues this one
            trail_top[0] = trail_size
            lit_top[0] = lit_size
            return PROBLEM_LIT_TRAIL_FULL
        prop_idx = buckets_pop(triggered_propagators, membership_offset)
        if prop_idx == -1:
            trail_top[0] = trail_size
            lit_top[0] = lit_size
            return PROBLEM_BOUND if state[unbound_idx] == 0 else PROBLEM_UNBOUND
        statistics[STATS_IDX_PROPAGATOR_FILTER_NB] += 1
        algorithm = algorithms[prop_idx]
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
        trail_size = trail_state_prefix(
            state, trail_log, trail_indices, mark, trail_size, prop_state_start, prop_state_hint
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
            trail_top[0] = trail_size
            lit_top[0] = lit_size
            return PROBLEM_INCONSISTENT
        if status == PROP_ENTAILMENT:
            statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB] += 1
            trail_size = record_entailment(entailed, trail_log, trail_indices, trail_size, entailed_idx, prop_idx)
        if reports_changes and status == PROP_CONSISTENCY and state[prop_state_hint] == 0:
            # the propagator wrote no domain, so update_domains would scan every variable only to find
            # no event and schedule nobody: skip it outright
            count_no_change(statistics, algorithm_stats)
            continue
        no_change, trail_size, lit_size = update_domains_with_reasons(
            prop_idx if flags & PROP_FLAG_IDEMPOTENT else -1,
            prop_var_start,
            prop_var_end,
            prop_domains,
            propagator_variables,
            state,
            trail_log,
            trail_indices,
            mark,
            trail_size,
            triggered_propagators,
            entailed,
            triggers,
            triggers_offsets,
            priorities,
            lit_trail,
            lit_last,
            lit_size,
            choice_point,
            prop_idx,
        )
        if no_change:
            count_no_change(statistics, algorithm_stats)


# always inlined, as update_domains is: lcg_algorithm is compiled without the reference-counting runtime
@njit(cache=True, inline="always")
def update_domains_with_reasons(
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
    lit_trail: NDArray,
    lit_last: NDArray,
    lit_size: int,
    level: int,
    prop_idx: int,
) -> tuple[bool, int, int]:
    """
    Applies a propagator's computed prop_domains as update_domains does, and adds a row to the implication trail for
    each bound that moves, with the propagator as its reason.

    The propagator call adds no row before its write-back, so the number of rows at the start of the write-back is the
    first row of the call: the REASON_POS of each of its rows.

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
    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param lit_size: the number of rows in use
    :type lit_size: int
    :param level: the current decision level
    :type level: int
    :param prop_idx: the propagator whose domains these are
    :type prop_idx: int

    :return: true iff no domain was changed, the new trail size and the new number of rows of the implication trail
    :rtype: Tuple[bool, int, int]
    """
    reason_pos = lit_size
    no_changes = True
    for var_idx in range(prop_var_end - prop_var_start):
        variable = propagator_variables[prop_var_start + var_idx]
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
                lit_size = lit_push_events(
                    lit_trail,
                    lit_last,
                    lit_size,
                    state,
                    variable,
                    events,
                    level,
                    REASON_PROPAGATOR,
                    prop_idx,
                    reason_pos,
                )
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
    return no_changes, trail_size, lit_size
