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
    bound = objective[OBJ_BOUND]
    value = objective[OBJ_VALUE]
    domain = domains_stk[top, objective[OBJ_VARIABLE]]
    was_bound = domain[MIN] == domain[MAX]
    if bound == MIN:
        new_value = max(value + 1, domain[MIN])
        if new_value == domain[MIN]:
            return EVENT_MASK_NONE
        domain[MIN] = new_value
        events = EVENT_MASK_MIN
    else:
        new_value = min(value - 1, domain[MAX])
        if new_value == domain[MAX]:
            return EVENT_MASK_NONE
        domain[MAX] = new_value
        events = EVENT_MASK_MAX
    if domain[MIN] > domain[MAX]:
        return -1
    if domain[MIN] == domain[MAX] and not was_bound:
        unbound_variable_nb_stk[top] -= 1
        events |= EVENT_MASK_GROUND
    return events


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
