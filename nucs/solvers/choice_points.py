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
    DECISION_EQ,
    DECISION_GT,
    DECISION_LE,
    DOM_UPDATE_EVENTS,
    DOM_UPDATE_VARIABLE,
    EVENT_MASK_GROUND,
    EVENT_MASK_MAX,
    EVENT_MASK_MIN,
    EVENT_MASK_NONE,
    MAX,
    MIN,
    OBJ_BOUND,
    OBJ_VALUE,
    OBJ_VARIABLE,
    STATS_IDX_SOLVER_BACKTRACK_NB,
)
from nucs.propagators.propagators import update_propagators


@njit(cache=True)
def cp_init(
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    domains_arr: NDArray,
    unbound_variable_nb: int,
) -> None:
    """
    Initializes the choice points.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size,
                             the following cells hold the indices of the entailed propagators in entailment order
    :type entailment_trail: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of unbound variable nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param domains_arr: the domains
    :type domains_arr: NDArray
    :param unbound_variable_nb: the number of unbound variables
    :type unbound_variable_nb: int
    """
    domains_stk[0] = domains_arr
    entailed_propagator_depths.fill(-1)
    domain_update_stk.fill(0)
    unbound_variable_nb_stk[0] = unbound_variable_nb
    entailment_trail[0] = stks_top[0] = 0


@njit(cache=True)
def cp_put(domains_stk: NDArray, unbound_variable_nb_stk: NDArray, top: int) -> None:
    """
    Adds a choice point to the stack of choice points.

    Entailment is monotonic within a branch, so descending to a deeper choice point requires no entailment
    bookkeeping: the depths recorded so far stay valid.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param top: the index of the top of the stacks
    :type top: int
    """
    # the two stacks are pushed in lockstep but stay separate on purpose. Folding the count into
    # domains_stk as an extra row would save one scalar store out of a whole-domains copy, gain nothing in
    # locality (top moves by one, so the live entry is always hot), put a non-domain value in an array whose
    # every other row is a domain, and change SIGN_DOM_HEURISTIC -- which every externally registered domain
    # heuristic is compiled against.
    domains_stk[top + 1] = domains_stk[top]  # copy the domains
    unbound_variable_nb_stk[top + 1] = unbound_variable_nb_stk[top]  # copy the number of unbound variables


@njit(cache=True, inline="always")
def tighten(
    domains: NDArray,
    unbound_variable_nb_stk: NDArray,
    top: int,
    variable: int,
    new_min: int,
    new_max: int,
) -> int:
    """
    Writes a variable's domain and returns the events the write triggers.

    This is the only place a backtrackable domain is written. Every other kind of domain write -- a
    propagator's filtering, a branching decision, the branch-and-bound clamp, a custom consistency
    algorithm's pruning -- routes through here, so the groundness test and the unbound-variable count
    stay consistent by construction rather than by convention, and a write barrier has a single place
    to live.

    Scheduling is deliberately left to the caller. The propagation loop schedules with a self-skip and a
    membership pre-test that update_propagators does not have, so folding the two loops together would
    either lose that specialization or drag six more parameters into the hottest function in the solver.

    The caller owns the wipeout test too: it is one comparison on a domain the caller is already holding,
    and the propagation loop can never wipe out (a propagator reports the inconsistency instead), so
    charging it for the check would be paying on the hot path for the cold callers.

    :param domains: the domains of the level being written
    :type domains: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param top: the index of the level being written
    :type top: int
    :param variable: the variable
    :type variable: int
    :param new_min: the new min of the domain
    :type new_min: int
    :param new_max: the new max of the domain
    :type new_max: int

    :return: the events triggered by the write, EVENT_MASK_NONE when it changed nothing
    :rtype: int
    """
    domain = domains[variable]
    was_bound = domain[MIN] == domain[MAX]
    events = EVENT_MASK_NONE
    if domain[MIN] != new_min:
        domain[MIN] = new_min
        events = EVENT_MASK_MIN
    if domain[MAX] != new_max:
        domain[MAX] = new_max
        events |= EVENT_MASK_MAX
    if events != EVENT_MASK_NONE and not was_bound and domain[MIN] == domain[MAX]:
        events |= EVENT_MASK_GROUND
        unbound_variable_nb_stk[top] -= 1
    return events


@njit(cache=True)
def branch(
    domains_stk: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    variable: int,
    kind: int,
    value: int,
) -> int:
    """
    Applies a decision: explores one branch and parks the alternatives for the search to resume.

    This is the whole of branching, in one place. The domain heuristics used to do it themselves, each
    repeating the same MIN/MAX/GROUND bookkeeping over two levels; they now only say where to split, and
    every kind of split lands here.

    The parked alternatives are written to the levels below the explored one, deepest first, so that
    backtracking meets them in that order. DECISION_EQ is the ternary case and parks two of them.

    An EQ value outside the domain is clamped into it rather than applied as written. min_cost returns -1
    when no value in the domain has a positive cost, and today that writes an out-of-domain [-1, -1] into
    the explored level and *widens* the parked one to [0, max]. Clamping keeps the split a partition of
    the domain, which is what makes the enumeration complete; failing the node instead would silently
    drop the assignments whose cost is zero.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param variable: the variable being branched on
    :type variable: int
    :param kind: the kind of split, one of DECISION_LE, DECISION_GT and DECISION_EQ
    :type kind: int
    :param value: the value the domain is split at
    :type value: int

    :return: the events triggered in the explored branch
    :rtype: int
    """
    top = stks_top[0]
    domain = domains_stk[top, variable]
    domain_min = domain[MIN]
    domain_max = domain[MAX]
    if kind == DECISION_EQ:
        value = min(max(value, domain_min), domain_max)
        # an EQ at either end of the domain has an empty alternative on that side, so it is a two-way
        # split: normalizing here is what keeps the ternary case to genuinely ternary splits
        if value == domain_min:
            kind = DECISION_LE
        elif value == domain_max:
            kind = DECISION_GT
            value = domain_max - 1
    if kind == DECISION_LE:
        cp_put(domains_stk, unbound_variable_nb_stk, top)
        parked = tighten(domains_stk[top], unbound_variable_nb_stk, top, variable, value + 1, domain_max)
        explored = tighten(domains_stk[top + 1], unbound_variable_nb_stk, top + 1, variable, domain_min, value)
        domain_update_stk[top, DOM_UPDATE_VARIABLE] = variable
        domain_update_stk[top, DOM_UPDATE_EVENTS] = parked
        stks_top[0] = top + 1
        return explored
    if kind == DECISION_GT:
        cp_put(domains_stk, unbound_variable_nb_stk, top)
        parked = tighten(domains_stk[top], unbound_variable_nb_stk, top, variable, domain_min, value)
        explored = tighten(domains_stk[top + 1], unbound_variable_nb_stk, top + 1, variable, value + 1, domain_max)
        domain_update_stk[top, DOM_UPDATE_VARIABLE] = variable
        domain_update_stk[top, DOM_UPDATE_EVENTS] = parked
        stks_top[0] = top + 1
        return explored
    cp_put(domains_stk, unbound_variable_nb_stk, top)
    cp_put(domains_stk, unbound_variable_nb_stk, top + 1)
    # the shallower alternative is the one resumed last
    parked_above = tighten(domains_stk[top], unbound_variable_nb_stk, top, variable, value + 1, domain_max)
    parked_below = tighten(domains_stk[top + 1], unbound_variable_nb_stk, top + 1, variable, domain_min, value - 1)
    explored = tighten(domains_stk[top + 2], unbound_variable_nb_stk, top + 2, variable, value, value)
    domain_update_stk[top + 1, DOM_UPDATE_VARIABLE] = domain_update_stk[top, DOM_UPDATE_VARIABLE] = variable
    domain_update_stk[top + 1, DOM_UPDATE_EVENTS] = parked_below
    domain_update_stk[top, DOM_UPDATE_EVENTS] = parked_above
    stks_top[0] = top + 2
    return explored


@njit(cache=True)
def unwind_entailment_trail(entailed_propagator_depths: NDArray, entailment_trail: NDArray, top: int) -> None:
    """
    Reactivates the propagators that were entailed below the current top.

    The trail is ordered by non-decreasing entailment depth, so it suffices to pop, from the top of the trail,
    every propagator whose entailment depth is strictly greater than the current top and reset it to active.

    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
    :param top: the index of the top of the stacks
    :type top: int
    """
    size = entailment_trail[0]
    while size > 0 and entailed_propagator_depths[entailment_trail[size]] > top:
        entailed_propagator_depths[entailment_trail[size]] = -1
        size -= 1
    entailment_trail[0] = size


@njit(cache=True)
def tighten_objective(
    domains_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    top: int,
    objective: NDArray,
) -> int:
    """
    Applies the current branch-and-bound bound to a level of the choice points.

    The bound is not backtrackable: it holds for the whole remaining search, so it is re-applied to each
    level as the search resumes it rather than written into every level when it is found. The tightening
    is monotone and idempotent, so re-applying it to an already-tightened level is a no-op.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param top: the index of the level to tighten
    :type top: int
    :param objective: the objective as a Numpy array of variable, bound and value
    :type objective: NDArray

    :return: the events triggered by the tightening, EVENT_MASK_NONE when it changed nothing,
             -1 when the level wipes out
    :rtype: int
    """
    variable = objective[OBJ_VARIABLE]
    value = objective[OBJ_VALUE]
    domains = domains_stk[top]
    domain = domains[variable]
    if objective[OBJ_BOUND] == MIN:
        events = tighten(domains, unbound_variable_nb_stk, top, variable, max(value + 1, domain[MIN]), domain[MAX])
    else:
        events = tighten(domains, unbound_variable_nb_stk, top, variable, domain[MIN], min(value - 1, domain[MAX]))
    return -1 if domain[MIN] > domain[MAX] else events


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    domains_stk: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
    domain_update_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    stks_top: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    priorities: NDArray,
    propagator_nb: int,
    objective: NDArray,
) -> bool:
    """
    Backtracks to the deepest choice point that can still hold a solution.

    When optimizing, the objective bound is re-applied to the level being resumed; a level it wipes out
    can no longer hold an improving solution, so the search keeps popping. This is why no level ever has
    to be pruned in advance.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
    :param domain_update_stk: the stack of domain updates
    :type domain_update_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :type triggered_propagators: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
    :param propagator_nb: the number of propagators
    :type propagator_nb: int
    :param objective: the objective as a Numpy array of variable, bound and value,
                      whose variable is -1 when not optimizing
    :type objective: NDArray

    :return: true iff it is possible to backtrack
    :rtype: bool
    """
    variable = objective[OBJ_VARIABLE]
    while stks_top[0] > 0:
        stks_top[0] -= 1
        top = stks_top[0]
        statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
        unwind_entailment_trail(entailed_propagator_depths, entailment_trail, top)
        domain_update = domain_update_stk[top]
        update_propagators(
            triggered_propagators,
            entailed_propagator_depths,
            triggers,
            triggers_offsets,
            priorities,
            propagator_nb,
            domain_update[DOM_UPDATE_VARIABLE],
            domain_update[DOM_UPDATE_EVENTS],
        )
        if variable < 0:
            return True
        events = tighten_objective(domains_stk, unbound_variable_nb_stk, top, objective)
        if events < 0:  # the level cannot hold an improving solution, keep popping
            continue
        if events != EVENT_MASK_NONE:
            update_propagators(
                triggered_propagators,
                entailed_propagator_depths,
                triggers,
                triggers_offsets,
                priorities,
                propagator_nb,
                variable,
                events,
            )
        return True
    return False


@njit(cache=True)
def fix_choice_point(
    domains_stk: NDArray,
    unbound_variable_nb_stk: NDArray,
    variable: int,
    value: int,
    bound: int,
) -> bool:
    """
    Fixes the domain of the variable being optimized in the top choice point.

    :param domains_stk: the stack of domains
    :type domains_stk: NDArray
    :param unbound_variable_nb_stk: the stack of the unbound variables nb
    :type unbound_variable_nb_stk: NDArray
    :param variable:  the variable being optimized
    :type variable: int
    :param value: the current optimal value for the variable
    :type value: int
    :param bound: the bound being optimized
    :type bound: int

    :return: true iff the resulting domain is non-empty
    :rtype: bool
    """
    domain = domains_stk[0, variable]
    domain[bound] = value + (1 if bound == MIN else -1)
    range_sz = domain[MAX] - domain[MIN]
    if range_sz < 0:
        return False
    if range_sz == 0:
        unbound_variable_nb_stk[0] -= 1
    return True
