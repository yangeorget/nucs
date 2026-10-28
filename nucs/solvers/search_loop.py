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
The compiled search loop of :class:`BacktrackSolver`: one call searches for one solution, and stops early when an
array it cannot grow runs out of room, when it must restart, or when it is interrupted.
"""

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN
from nucs.numba_helper import ConsistencyAlgorithmFunctions, DomainHeuristicFunctions, VariableHeuristicFunctions
from nucs.problems.problem import PROBLEM_BOUND, PROBLEM_UNBOUND
from nucs.propagators.propagators import update_propagators
from nucs.solvers.choice_points import backtrack, branch
from nucs.solvers.solver import get_solution
from nucs.statistics import STATS_IDX_SOLUTION_NB, STATS_IDX_SOLVER_CHOICE_DEPTH, STATS_IDX_SOLVER_CHOICE_NB

# Why run_search_loop returned.
# The trail and the choice point stack are caller-allocated, so they cannot grow inside @njit. Rather than
# sizing them for a worst case that never happens -- depth x (2 x domain_nb + 1) entries, which would
# give back the memory this representation wins -- the search stops and says which one is full, and the
# solver grows it and resumes. Nothing of the search is lost: the state, the trail marks and the
# positions all stay valid across the reallocation.
SOLVER_RUNNING = 0  # nothing filled up: the search returned a solution or exhausted itself
SOLVER_TRAIL_FULL = 1  # the search stopped because the trail needs more room, not because it is over
SOLVER_CHOICE_POINTS_FULL = 2  # likewise for the stack of choice points
SOLVER_INTERRUPTED = 3  # the search stopped because interrupt() asked it to, from outside the compiled loop
SOLVER_RESTART = 4  # the descent reached its failure limit: the solver restarts from the root and resumes

# The cells of search_control, the state of the restarts and of last-conflict that the compiled loop reads and writes.
# It is solver state, not choice-point state: nothing about it is trailed.
SEARCH_CONTROL_RESTART_LIMIT = 0  # the failures after which the descent restarts, -1 for no restart
SEARCH_CONTROL_FAILURE_NB = 1  # the failures since the search last started from the root
SEARCH_CONTROL_LAST_CONFLICT = 2  # 1 when last-conflict is on, 0 otherwise
SEARCH_CONTROL_CONFLICT_VARIABLE = 3  # the variable of the last decision that led to a failure, -1 when none
SEARCH_CONTROL_DECISION_VARIABLE = 4  # the variable of the last decision, -1 once a refutation follows it
SEARCH_CONTROL_WIDTH = 5


@njit(cache=True, nogil=True)
def run_search_loop(
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
    entailed: NDArray,
    trail: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    choice_point_stk: NDArray,
    choice_point_top: NDArray,
    triggered_propagators: NDArray,
    consistency_alg_fcts: ConsistencyAlgorithmFunctions,
    decision_variables: NDArray,
    decision_variables_offsets: NDArray,
    var_heuristic_fcts: VariableHeuristicFunctions,
    var_heuristic_params: NDArray,
    var_heuristic_params_offsets: NDArray,
    var_heuristic_params_shapes: NDArray,
    dom_heuristic_fcts: DomainHeuristicFunctions,
    dom_heuristic_params: NDArray,
    dom_heuristic_params_offsets: NDArray,
    dom_heuristic_params_shapes: NDArray,
    compute_domains_addrs: NDArray,
    domain_buffer: NDArray,
    algorithm_flags: NDArray,
    bb: NDArray,
    trail_headroom: int,
    interruption: NDArray,
    propagator_weights: NDArray,
    search_control: NDArray,
    variable_searches: NDArray,
) -> tuple[int, NDArray | None]:
    """
    Searches for one solution, stopping early when an array it cannot grow runs out of room.

    It returns before the end of the search because the trail and the choice point stack belong to the
    caller: when either fills up, the returned status names it and no solution comes back, for _next_solution
    to grow that array and call again. Nothing of the search is lost in between.

    The status is returned rather than written into a one-cell array because nothing inside @njit reads
    it -- it is the reason the loop stopped, and only the Python caller acts on it. A solution alone
    cannot carry that reason: None means both "the search is over" and "grow an array and call me again".

    Expects the propagation queue to already hold the propagators that need to run: the callers enqueue
    all the propagators (buckets_init) before the first call, and rely on backtrack to schedule the
    propagators affected by a parked alternative, or by the objective bound it re-applies, between subsequent
    calls.

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
    :param state: all the backtrackable state: the domain bounds followed by the unbound-variable count
    :type state: NDArray
    :param domains: the current domains, a (domain_nb, 2) view of the head of state
    :type domains: NDArray
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param trail: the undo log of (cell index, old value) pairs
    :type trail: NDArray
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
    :param consistency_alg_fcts: a 1-element list holding the consistency algorithm function
    :type consistency_alg_fcts: ConsistencyAlgorithmFunctions
    :param decision_variables: the concatenation of the per-search decision variable arrays
    :type decision_variables: NDArray
    :param decision_variables_offsets: the CSR offsets delimiting each search's slice of decision_variables
    :type decision_variables_offsets: NDArray
    :param var_heuristic_fcts: the typed list of variable heuristic functions, one per search
    :type var_heuristic_fcts: VariableHeuristicFunctions
    :param var_heuristic_params: the flattened concatenation of the per-search variable heuristic parameter arrays
    :type var_heuristic_params: NDArray
    :param var_heuristic_params_offsets: the CSR offsets delimiting each search's slice of var_heuristic_params
    :type var_heuristic_params_offsets: NDArray
    :param var_heuristic_params_shapes: the 2d shape of each search's variable heuristic parameter array
    :type var_heuristic_params_shapes: NDArray
    :param dom_heuristic_fcts: the typed list of domain heuristic functions, one per search
    :type dom_heuristic_fcts: DomainHeuristicFunctions
    :param dom_heuristic_params: the flattened concatenation of the per-search domain heuristic parameter arrays
    :type dom_heuristic_params: NDArray
    :param dom_heuristic_params_offsets: the CSR offsets delimiting each search's slice of dom_heuristic_params
    :type dom_heuristic_params_offsets: NDArray
    :param dom_heuristic_params_shapes: the 2d shape of each search's domain heuristic parameter array
    :type dom_heuristic_params_shapes: NDArray
    :param compute_domains_addrs: the compiled compute_domains address of each algorithm, resolved once at solver init
    :type compute_domains_addrs: NDArray
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray
    :param algorithm_flags: the PROP_FLAG_* properties of each algorithm, packed into one word and indexed by
                          algorithm rather than by propagator
    :type algorithm_flags: NDArray
    :param bb: the branch-and-bound bound as a Numpy array of variable, bound and value,
               whose variable is -1 when not optimizing
    :type bb: NDArray
    :param trail_headroom: the trail entries any one node of the search can need
    :type trail_headroom: int
    :param interruption: the one-cell array of an Interrupter, non-zero once the search is asked to stop
    :type interruption: NDArray
    :param propagator_weights: the failure weight of each propagator, followed by the increment and its growth
                               (see nucs.solvers.weights)
    :type propagator_weights: NDArray
    :param search_control: the state of the restarts and of last-conflict, see SEARCH_CONTROL_*
    :type search_control: NDArray
    :param variable_searches: the search that owns each variable, -1 for a variable no search branches on
    :type variable_searches: NDArray

    :return: why the loop returned, and the solution when it found one
    :rtype: Tuple[int, Optional[NDArray]]
    """
    consistency_alg_fct = consistency_alg_fcts[0]
    nb_searches = len(decision_variables_offsets) - 1
    max_choice_point = len(choice_point_stk) - 3  # a ternary split pushes two choice points and marks a third
    while True:
        # checked at every node and before the state is touched, like the two checks below
        if interruption[0] != 0:
            return SOLVER_INTERRUPTED, None
        # checked before the filtering, so that the state the solver restarts from is not a half-done one
        if 0 <= search_control[SEARCH_CONTROL_RESTART_LIMIT] <= search_control[SEARCH_CONTROL_FAILURE_NB]:
            return SOLVER_RESTART, None
        # the arrays are caller-allocated, so the search stops for the solver to grow one rather than
        # overrun it silently -- with boundscheck off, the overrun is what would otherwise happen
        if trail_top[0] + trail_headroom > len(trail):
            return SOLVER_TRAIL_FULL, None
        if choice_point_top[0] > max_choice_point:
            return SOLVER_CHOICE_POINTS_FULL, None
        problem_status = consistency_alg_fct(
            statistics,
            propagator_weights,
            algorithm_flags,
            algorithms,
            priorities,
            offsets,
            propagator_variables,
            propagator_parameters,
            triggers,
            triggers_offsets,
            state,
            domains,
            entailed,
            trail,
            trail_top,
            trail_indices,
            choice_point_stk,
            choice_point_top,
            triggered_propagators,
            compute_domains_addrs,
            domain_buffer,
        )
        if problem_status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLUTION_NB] += 1
            return SOLVER_RUNNING, get_solution(domains)
        branched = False
        if problem_status == PROBLEM_UNBOUND:
            # sequential search: the first search that still has an unbound decision variable owns the
            # decision and branches with its own variable and domain heuristics
            conflict_variable = search_control[SEARCH_CONTROL_CONFLICT_VARIABLE]
            for search_idx in range(nb_searches):
                # last-conflict: the variable of the last refuted decision, while it is unbound, takes the decision
                # from the heuristic of the search that owns it. The searches before this one have no unbound
                # variable left, or the loop would not be here, so this keeps the order of the searches.
                if (
                    conflict_variable >= 0
                    and variable_searches[conflict_variable] == search_idx
                    and domains[conflict_variable, DOMAIN_MIN] < domains[conflict_variable, DOMAIN_MAX]
                ):
                    variable = conflict_variable
                else:
                    variable = var_heuristic_fcts[search_idx](
                        decision_variables[
                            decision_variables_offsets[search_idx] : decision_variables_offsets[search_idx + 1]
                        ],
                        domains,
                        entailed,
                        offsets,
                        propagator_variables,
                        triggers,
                        triggers_offsets,
                        propagator_weights,
                        var_heuristic_params[
                            var_heuristic_params_offsets[search_idx] : var_heuristic_params_offsets[search_idx + 1]
                        ].reshape(
                            var_heuristic_params_shapes[search_idx, 0], var_heuristic_params_shapes[search_idx, 1]
                        ),
                    )
                if variable != -1:
                    # the heuristic only says where to split; branch owns the two choice points it takes to do so
                    kind, value = dom_heuristic_fcts[search_idx](
                        domains,
                        variable,
                        dom_heuristic_params[
                            dom_heuristic_params_offsets[search_idx] : dom_heuristic_params_offsets[search_idx + 1]
                        ].reshape(
                            dom_heuristic_params_shapes[search_idx, 0], dom_heuristic_params_shapes[search_idx, 1]
                        ),
                    )
                    events = branch(
                        state,
                        trail,
                        trail_top,
                        trail_indices,
                        choice_point_stk,
                        choice_point_top,
                        variable,
                        kind,
                        value,
                    )
                    choice_point = choice_point_top[0]
                    update_propagators(
                        triggered_propagators, entailed, triggers, triggers_offsets, priorities, variable, events, -1
                    )
                    statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
                    if search_control[SEARCH_CONTROL_LAST_CONFLICT]:
                        search_control[SEARCH_CONTROL_DECISION_VARIABLE] = variable
                    statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = max(
                        statistics[STATS_IDX_SOLVER_CHOICE_DEPTH], choice_point
                    )
                    branched = True
                    break
        # either the problem is inconsistent, or no search can claim a variable although variables remain
        # unbound -- a choice point whose domains admit no assignment. Both are dead ends: backtrack.
        if not branched:
            search_control[SEARCH_CONTROL_FAILURE_NB] += 1
            # last-conflict (Lecoutre et al. 2009): the conflict variable is that of the last decision that led to a
            # failure. A failure that follows a refutation keeps the conflict variable: after x fails on all its
            # values the search refutes an earlier decision on y, and x must then be tried first, to test y.
            if search_control[SEARCH_CONTROL_DECISION_VARIABLE] >= 0:
                search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = search_control[SEARCH_CONTROL_DECISION_VARIABLE]
                search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
            if not backtrack(
                statistics,
                state,
                trail,
                trail_top,
                trail_indices,
                choice_point_stk,
                choice_point_top,
                entailed,
                triggered_propagators,
                triggers,
                triggers_offsets,
                priorities,
                bb,
            ):
                return SOLVER_RUNNING, None
