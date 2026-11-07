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
BacktrackSolver and its compiled search loop: one call of the loop searches for one solution, and stops early when
an array it cannot grow runs out of room, when it must restart, or when it is interrupted.
"""

import logging
from collections.abc import Sequence

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import DOMAIN_MAX, DOMAIN_MIN, LOG_LEVEL_INFO
from nucs.numba_helper import (
    ConsistencyAlgorithmFunctions,
    DomainHeuristicFunctions,
    VariableHeuristicFunctions,
    build_function_ptrs,
)
from nucs.problems.problem import PROBLEM_BOUND, PROBLEM_UNBOUND, Problem
from nucs.solvers.backtrack.consistency_algorithms import (
    CONSISTENCY_ALG_BC,
    CONSISTENCY_ALG_FCTS,
    SIGN_CONSISTENCY_ALG,
    get_consistency_algorithm_name,
)
from nucs.solvers.choice_points import backtrack, branch
from nucs.solvers.interrupter import INTERRUPTION_DEADLINE, INTERRUPTION_EXTERNAL
from nucs.solvers.propagation_steps import update_propagators
from nucs.solvers.restart_policies import RESTART_POLICY_NEVER, RestartPolicy
from nucs.solvers.searches import SEARCHES_DEFAULT, Search
from nucs.solvers.tree_search_solver import (
    SEARCH_CONTROL_CONFLICT_VARIABLE,
    SEARCH_CONTROL_DECISION_VARIABLE,
    SEARCH_CONTROL_FAILURE_NB,
    SEARCH_CONTROL_LAST_CONFLICT,
    SEARCH_CONTROL_RESTART_LIMIT,
    SOLVER_CHOICE_POINTS_FULL,
    SOLVER_INTERRUPTED,
    SOLVER_RESTART,
    SOLVER_RUNNING,
    SOLVER_UTRAIL_FULL,
    TreeSearchSolver,
    get_solution,
)
from nucs.statistics import STATS_IDX_SOLUTION_NB, STATS_IDX_SOLVER_CHOICE_DEPTH, STATS_IDX_SOLVER_CHOICE_NB

logger = logging.getLogger(__name__)


class BacktrackSolver(TreeSearchSolver):
    """
    A solver relying on a backtracking mechanism, with a propagation that a consistency algorithm gives.
    """

    def __init__(
        self,
        problem: Problem,
        consistency_algorithm: int = CONSISTENCY_ALG_BC,
        searches: Sequence[Search] = SEARCHES_DEFAULT,
        log_level: str = LOG_LEVEL_INFO,
        weight_decay: float = 1.0,
        restart_policy: RestartPolicy = RESTART_POLICY_NEVER,
        last_conflict: bool = False,
    ):
        """
        Initializes the solver.

        :param problem: the problem to be solved
        :type problem: Problem
        :param consistency_algorithm: the consistency algorithm, defaults to bound consistency
        :type consistency_algorithm: int
        :param searches: an ordered sequence of searches defining a sequential search, each with its decision
                         variables, its variable and domain heuristics and their parameters; defaults to
                         SEARCHES_DEFAULT, one search that branches on every variable with the default heuristics of
                         Search. The union of the searches' decision variables should cover every branchable variable.
        :type searches: Sequence[Search]
        :param log_level: the log level,
                          defaults to INFO
        :type log_level: str
        :param weight_decay: in (0, 1], how much the failure weight of a propagator keeps of a failure after each
                             later failure; 1 counts all the failures the same (dom/wdeg), less than 1 prefers the
                             recent ones (AFC), defaults to 1
        :type weight_decay: float
        :param restart_policy: after how many failures each descent restarts from the root
                               (nucs.solvers.restart_policies), defaults to no restart. When enumerating, the restarts
                               stop at the first solution, so that no solution is found twice.
        :type restart_policy: RestartPolicy
        :param last_conflict: whether the search branches first on the variable of the last refuted decision, as long
                              as it is unbound (last-conflict reasoning), defaults to False
        :type last_conflict: bool
        """
        super().__init__(problem, searches, log_level, weight_decay, restart_policy, last_conflict)
        logger.info(
            f"BacktrackSolver uses consistency algorithm {get_consistency_algorithm_name(consistency_algorithm)}"
        )
        # a typed list of 1 function under the JIT, a plain Python list otherwise (see build_function_ptrs)
        self.consistency_alg_fcts = build_function_ptrs(
            [CONSISTENCY_ALG_FCTS[consistency_algorithm]], SIGN_CONSISTENCY_ALG
        )

    def _backtrack(self) -> bool:
        """
        Backtracks by forwarding the solver state to the jitted backtrack.

        :return: true iff it was possible to backtrack
        :rtype: bool
        """
        return backtrack(
            self.statistics,
            self.state,
            self.utrail_log,
            self.utrail_top,
            self.utrail_indices,
            self.choice_point_stk,
            self.choice_point_top,
            self.entailed,
            self.triggered_propagators,
            self.problem.triggers,
            self.problem.triggers_offsets,
            self.problem.priorities,
            self.bb,
        )

    def _run_search_loop(self) -> tuple[int, NDArray | None]:
        """
        Runs the compiled search loop until it returns, by forwarding the solver state to the jitted run_search_loop.

        :return: why the loop returned, and the solution when it found one
        :rtype: Tuple[int, Optional[NDArray]]
        """
        return run_search_loop(
            self.statistics,
            self.problem.algorithms,
            self.problem.priorities,
            self.problem.offsets,
            self.problem.propagator_variables,
            self.problem.propagator_parameters,
            self.problem.triggers,
            self.problem.triggers_offsets,
            self.state,
            self.domains,
            self.entailed,
            self.utrail_log,
            self.utrail_top,
            self.utrail_indices,
            self.choice_point_stk,
            self.choice_point_top,
            self.triggered_propagators,
            self.consistency_alg_fcts,
            self.flat_searches.decision_variables,
            self.flat_searches.decision_variables_offsets,
            self.var_heuristic_fcts,
            self.flat_searches.var_heuristic_params,
            self.flat_searches.var_heuristic_params_offsets,
            self.flat_searches.var_heuristic_params_shapes,
            self.dom_heuristic_fcts,
            self.flat_searches.dom_heuristic_params,
            self.flat_searches.dom_heuristic_params_offsets,
            self.flat_searches.dom_heuristic_params_shapes,
            self.compute_domains_addrs,
            self.domain_buffer,
            self.problem.algorithm_flags,
            self.bb,
            self.utrail_headroom,
            self.interrupter.cells,
            self.propagator_weights,
            self.search_control,
            self.flat_searches.variable_searches,
        )


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
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
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
    utrail_headroom: int,
    interruption: NDArray,
    propagator_weights: NDArray,
    search_control: NDArray,
    variable_searches: NDArray,
) -> tuple[int, NDArray | None]:
    """
    Searches for one solution, stopping early when an array it cannot grow runs out of room.

    It returns before the end of the search because the undo trail and the choice point stack belong to the
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
    :param utrail_log: the undo log of (cell index, old value) pairs
    :type utrail_log: NDArray
    :param utrail_top: the undo trail size as a Numpy array
    :type utrail_top: NDArray
    :param utrail_indices: the index of the last undo trail entry per positionally guarded cell
    :type utrail_indices: NDArray
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
    :param utrail_headroom: the undo trail entries any one node of the search can need
    :type utrail_headroom: int
    :param interruption: the two flags of an Interrupter, one of them set once the search is asked to stop
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
        if interruption[INTERRUPTION_EXTERNAL] | interruption[INTERRUPTION_DEADLINE]:
            return SOLVER_INTERRUPTED, None
        # checked before the filtering, so that the state the solver restarts from is not a half-done one
        if 0 <= search_control[SEARCH_CONTROL_RESTART_LIMIT] <= search_control[SEARCH_CONTROL_FAILURE_NB]:
            return SOLVER_RESTART, None
        # the arrays are caller-allocated, so the search stops for the solver to grow one rather than
        # overrun it silently -- with boundscheck off, the overrun is what would otherwise happen
        if utrail_top[0] + utrail_headroom > len(utrail_log):
            return SOLVER_UTRAIL_FULL, None
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
            utrail_log,
            utrail_top,
            utrail_indices,
            choice_point_stk,
            choice_point_top,
            triggered_propagators,
            compute_domains_addrs,
            domain_buffer,
        )
        if problem_status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLUTION_NB] += 1
            return SOLVER_RUNNING, get_solution(domains)
        # the variable to branch on, -1 when there is none, and the search that owns it; search_idx is set before
        # the loop so that it has a value on every path
        search_idx = 0
        variable = -1
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
                    break
        # either the problem is inconsistent, or no search can claim a variable although variables remain
        # unbound -- a choice point whose domains admit no assignment. Both are dead ends: backtrack.
        if variable == -1:
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
                utrail_log,
                utrail_top,
                utrail_indices,
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
            continue
        # the heuristic only says where to split; branch owns the two choice points it takes to do so
        kind, value = dom_heuristic_fcts[search_idx](
            domains,
            variable,
            dom_heuristic_params[
                dom_heuristic_params_offsets[search_idx] : dom_heuristic_params_offsets[search_idx + 1]
            ].reshape(dom_heuristic_params_shapes[search_idx, 0], dom_heuristic_params_shapes[search_idx, 1]),
        )
        events = branch(
            state,
            utrail_log,
            utrail_top,
            utrail_indices,
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
        statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = max(statistics[STATS_IDX_SOLVER_CHOICE_DEPTH], choice_point)
