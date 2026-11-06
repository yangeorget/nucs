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
The solver of lazy clause generation (design/proposals/lcg.md) and its compiled search loop.
"""

import logging
from collections.abc import Sequence

from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import BB_BOUND, BB_VALUE, BB_VARIABLE, DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_NONE, LOG_LEVEL_INFO
from nucs.numba_helper import DomainHeuristicFunctions, VariableHeuristicFunctions
from nucs.problems.problem import PROBLEM_BOUND, PROBLEM_UNBOUND, Problem
from nucs.solvers.choice_points import (
    CHOICE_POINT_BOUND,
    CHOICE_POINT_TRAIL_MARK,
    CHOICE_POINT_VALUE,
    CHOICE_POINT_VARIABLE,
    bounds_after_tightening,
    branch,
    tighten_objective,
)
from nucs.solvers.interrupter import INTERRUPTION_DEADLINE, INTERRUPTION_EXTERNAL
from nucs.solvers.lcg.implication_trail import (
    LIT_TRAIL_VALID,
    REASON_DECISION,
    REASON_ROOT,
    allocate_implication_trail,
    lit_pop,
    lit_push_events,
    lit_trail_check,
    lit_trail_root,
)
from nucs.solvers.lcg.lcg_algorithm import PROBLEM_LIT_TRAIL_FULL, lcg_algorithm
from nucs.solvers.propagation_steps import update_propagators
from nucs.solvers.restart_policies import RESTART_POLICY_NEVER, RestartPolicy
from nucs.solvers.searches import SEARCHES_DEFAULT, Search
from nucs.solvers.state import tighten_at, trail_undo
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
    SOLVER_TRAIL_FULL,
    TreeSearchSolver,
    double_rows,
    get_solution,
    stack_height,
)
from nucs.statistics import (
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
)

logger = logging.getLogger(__name__)

# Why the search loop of LcgSolver returned, in addition to the SOLVER_* statuses of tree_search_solver.py. LcgSolver
# handles these two itself, in _run_search_loop: TreeSearchSolver never sees them.
SOLVER_LIT_TRAIL_FULL = 5  # the implication trail needs more room
SOLVER_LIT_TRAIL_INVALID = 6  # in debug mode, the implication trail does not agree with the state


class LcgSolver(TreeSearchSolver):
    """
    A solver that will learn clauses from its failures (lazy clause generation).

    Its propagation is always bound consistency, and no parameter selects another one: the conflict analysis needs a
    reason for each change of a bound, and a custom consistency algorithm gives none.

    At this step (design/proposals/lcg.md, 7.0, step 5), it records each bound change on the path, with its decision
    level and its reason, on the implication trail (nucs.solvers.lcg.implication_trail). The search does not use the
    trail yet: it is the search of BacktrackSolver, and every counter is the same.
    """

    def __init__(
        self,
        problem: Problem,
        searches: Sequence[Search] = SEARCHES_DEFAULT,
        log_level: str = LOG_LEVEL_INFO,
        weight_decay: float = 1.0,
        restart_policy: RestartPolicy = RESTART_POLICY_NEVER,
        last_conflict: bool = False,
        debug: bool = False,
    ):
        """
        Initializes the solver.

        :param problem: the problem to be solved
        :type problem: Problem
        :param searches: an ordered sequence of searches defining a sequential search, as for BacktrackSolver
        :type searches: Sequence[Search]
        :param log_level: the log level,
                          defaults to INFO
        :type log_level: str
        :param weight_decay: in (0, 1], how much the failure weight of a propagator keeps of a failure after each
                             later failure, as for BacktrackSolver, defaults to 1
        :type weight_decay: float
        :param restart_policy: after how many failures each descent restarts from the root, as for BacktrackSolver,
                               defaults to no restart
        :type restart_policy: RestartPolicy
        :param last_conflict: whether the search branches first on the variable of the last refuted decision, as for
                              BacktrackSolver, defaults to False
        :type last_conflict: bool
        :param debug: whether the search replays the implication trail from the root at each node and compares the
                      result with the state; an AssertionError tells the first row that does not agree. This is slow:
                      it is for the tests. Defaults to False
        :type debug: bool
        """
        # allocated before TreeSearchSolver.__init__, because it calls _record_root
        self.lit_trail, self.lit_top, self.lit_last, self.level_start = allocate_implication_trail(
            problem.domain_nb, stack_height(problem.domain_nb)
        )
        self.debug = debug
        super().__init__(problem, searches, log_level, weight_decay, restart_policy, last_conflict)
        logger.info(f"The implication trail starts at {len(self.lit_trail)} rows and grows when it runs out")
        logger.info(f"LcgSolver checks the implication trail at each node: {'yes' if debug else 'no'}")

    def _record_root(self) -> None:
        """
        Builds the implication trail again for the root that the solver has just set.
        """
        lit_trail_root(
            self.state, self.problem.initial_domains, self.lit_trail, self.lit_top, self.lit_last, self.level_start
        )

    def _backtrack(self) -> bool:
        """
        Backtracks by forwarding the solver state to the jitted backtrack of LcgSolver.

        :return: true iff it was possible to backtrack
        :rtype: bool
        """
        return backtrack(
            self.statistics,
            self.state,
            self.trail_log,
            self.trail_top,
            self.trail_indices,
            self.choice_point_stk,
            self.choice_point_top,
            self.entailed,
            self.triggered_propagators,
            self.problem.triggers,
            self.problem.triggers_offsets,
            self.problem.priorities,
            self.bb,
            self.lit_trail,
            self.lit_top,
            self.lit_last,
            self.level_start,
        )

    def _run_search_loop(self) -> tuple[int, NDArray | None]:
        """
        Runs the compiled search loop of LcgSolver until it returns, by forwarding the solver state to it.

        It grows the implication trail itself when the loop asks for it, and calls the loop again. It also keeps
        level_start as long as the stack of choice points, which TreeSearchSolver grows.

        :return: why the loop returned, and the solution when it found one
        :rtype: Tuple[int, Optional[NDArray]]
        """
        while True:
            while len(self.level_start) < len(self.choice_point_stk):
                self.level_start = double_rows(self.level_start)
            status, solution = run_search_loop(
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
                self.trail_log,
                self.trail_top,
                self.trail_indices,
                self.choice_point_stk,
                self.choice_point_top,
                self.triggered_propagators,
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
                self.trail_headroom,
                self.interrupter.cells,
                self.propagator_weights,
                self.search_control,
                self.flat_searches.variable_searches,
                self.lit_trail,
                self.lit_top,
                self.lit_last,
                self.level_start,
                self.problem.initial_domains,
                self.debug,
            )
            if status == SOLVER_LIT_TRAIL_FULL:
                self.lit_trail = double_rows(self.lit_trail)
                logger.info(f"The implication trail grew to {len(self.lit_trail)} rows")
            elif status == SOLVER_LIT_TRAIL_INVALID:
                row = lit_trail_check(
                    self.state,
                    self.problem.initial_domains,
                    self.lit_trail,
                    self.lit_top,
                    self.lit_last,
                    self.level_start,
                    self.choice_point_top[0],
                )
                raise AssertionError(
                    f"The implication trail does not agree with the state at row {row} of {self.lit_top[0]}"
                )
            else:
                return status, solution


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
    lit_trail: NDArray,
    lit_top: NDArray,
    lit_last: NDArray,
    level_start: NDArray,
    initial_domains: NDArray,
    debug: bool,
) -> tuple[int, NDArray | None]:
    """
    Searches for one solution, as the run_search_loop of BacktrackSolver does: the search loop of LcgSolver.

    It is a copy of the run_search_loop of BacktrackSolver with three changes (design/proposals/lcg.md, 7.0, step 5):

    - it calls lcg_algorithm directly, where that loop calls the consistency algorithm of the solver; lcg_algorithm
      records the changes that the propagators make on the implication trail;
    - a decision opens its levels in level_start, and adds its rows with REASON_DECISION;
    - it backtracks with the backtrack of LcgSolver, which also pops the implication trail.

    The level of a row is the height of the stack of choice points when the row is written. A ternary decision
    pushes two choice points, so its first level holds no row.

    It stops early when an array it cannot grow runs out of room.

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
    :param interruption: the two flags of an Interrupter, one of them set once the search is asked to stop
    :type interruption: NDArray
    :param propagator_weights: the failure weight of each propagator, followed by the increment and its growth
                               (see nucs.solvers.weights)
    :type propagator_weights: NDArray
    :param search_control: the state of the restarts and of last-conflict, see SEARCH_CONTROL_*
    :type search_control: NDArray
    :param variable_searches: the search that owns each variable, -1 for a variable no search branches on
    :type variable_searches: NDArray
    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_top: the number of rows in use, as a one-cell array
    :type lit_top: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param level_start: the first row of each decision level
    :type level_start: NDArray
    :param initial_domains: the domains of the problem before any propagation, for the debug check
    :type initial_domains: NDArray
    :param debug: whether to replay the implication trail and compare it with the state at each node
    :type debug: bool

    :return: why the loop returned, and the solution when it found one
    :rtype: Tuple[int, Optional[NDArray]]
    """
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
        if trail_top[0] + trail_headroom > len(trail):
            return SOLVER_TRAIL_FULL, None
        if choice_point_top[0] > max_choice_point:
            return SOLVER_CHOICE_POINTS_FULL, None
        # no test of the room of the implication trail here: lcg_algorithm makes sure, before each propagator call,
        # that the rows of the call and those of the next decision or backtrack fit
        if debug and not is_lit_trail_valid(
            state, initial_domains, lit_trail, lit_top, lit_last, level_start, choice_point_top
        ):
            return SOLVER_LIT_TRAIL_INVALID, None
        problem_status = lcg_algorithm(
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
            lit_trail,
            lit_top,
            lit_last,
        )
        if problem_status == PROBLEM_LIT_TRAIL_FULL:
            return SOLVER_LIT_TRAIL_FULL, None
        if debug and not is_lit_trail_valid(
            state, initial_domains, lit_trail, lit_top, lit_last, level_start, choice_point_top
        ):
            return SOLVER_LIT_TRAIL_INVALID, None
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
                lit_trail,
                lit_top,
                lit_last,
                level_start,
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
        level = choice_point_top[0]
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
        # the decision opens one level, or two for a ternary split; its rows are on the deepest one
        lit_size = lit_top[0]
        for new_level in range(level + 1, choice_point + 1):
            level_start[new_level] = lit_size
        lit_top[0] = lit_push_events(
            lit_trail, lit_last, lit_size, state, variable, events, choice_point, REASON_DECISION, -1, -1
        )
        update_propagators(
            triggered_propagators, entailed, triggers, triggers_offsets, priorities, variable, events, -1
        )
        statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
        if search_control[SEARCH_CONTROL_LAST_CONFLICT]:
            search_control[SEARCH_CONTROL_DECISION_VARIABLE] = variable
        statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = max(statistics[STATS_IDX_SOLVER_CHOICE_DEPTH], choice_point)


@njit(cache=True, inline="always")
def is_lit_trail_valid(
    state: NDArray,
    initial_domains: NDArray,
    lit_trail: NDArray,
    lit_top: NDArray,
    lit_last: NDArray,
    level_start: NDArray,
    choice_point_top: NDArray,
) -> bool:
    """
    Returns whether the implication trail agrees with the state, at the current level: the debug check.

    :param state: all the backtrackable state
    :type state: NDArray
    :param initial_domains: the domains of the problem before any propagation
    :type initial_domains: NDArray
    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_top: the number of rows in use, as a one-cell array
    :type lit_top: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param level_start: the first row of each decision level
    :type level_start: NDArray
    :param choice_point_top: the index of the top of the choice points as a Numpy array, the current level
    :type choice_point_top: NDArray

    :return: true iff the implication trail agrees with the state
    :rtype: bool
    """
    return (
        lit_trail_check(state, initial_domains, lit_trail, lit_top, lit_last, level_start, choice_point_top[0])
        == LIT_TRAIL_VALID
    )


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    state: NDArray,
    trail_log: NDArray,
    trail_top: NDArray,
    trail_indices: NDArray,
    choice_point_stk: NDArray,
    choice_point_top: NDArray,
    entailed: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    priorities: NDArray,
    bb: NDArray,
    lit_trail: NDArray,
    lit_top: NDArray,
    lit_last: NDArray,
    level_start: NDArray,
) -> bool:
    """
    Backtracks to the deepest choice point that can still hold a solution, as the backtrack of BacktrackSolver
    (nucs.solvers.choice_points) does, and keeps the implication trail in step.

    It is a copy of that backtrack with two changes: popping a choice point also pops the rows of the levels above
    it, and the parked alternative and the objective bound add their rows, at the level of the choice point that the
    search resumes. The alternative gets REASON_DECISION: at this step it is a decision of the search, as in
    BacktrackSolver. The objective bound gets REASON_ROOT: it holds for the rest of the search.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param state: all the backtrackable state
    :type state: NDArray
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
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :type triggered_propagators: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param priorities: the propagation queue bucket priorities indexed by propagators
    :type priorities: NDArray
    :param bb: the branch-and-bound bound as a Numpy array of variable, bound and value,
               whose variable is -1 when not optimizing
    :type bb: NDArray
    :param lit_trail: the implication trail
    :type lit_trail: NDArray
    :param lit_top: the number of rows in use, as a one-cell array
    :type lit_top: NDArray
    :param lit_last: the last row of each bound, -1 when the bound has its root value
    :type lit_last: NDArray
    :param level_start: the first row of each decision level
    :type level_start: NDArray

    :return: true iff it is possible to backtrack
    :rtype: bool
    """
    optimized_variable = bb[BB_VARIABLE]
    while choice_point_top[0] > 0:
        choice_point_top[0] -= 1
        choice_point = choice_point_top[0]
        statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
        mark = choice_point_stk[choice_point, CHOICE_POINT_TRAIL_MARK]
        trail_undo(state, trail_log, trail_indices, trail_top, mark)
        # the level of this choice point stays: the rows of the levels above it go
        lit_pop(lit_trail, lit_top, lit_last, level_start[choice_point + 1])
        variable = choice_point_stk[choice_point, CHOICE_POINT_VARIABLE]
        cell_idx = variable << 1
        new_min, new_max = bounds_after_tightening(
            state[cell_idx],
            state[cell_idx | 1],
            choice_point_stk[choice_point, CHOICE_POINT_BOUND],
            choice_point_stk[choice_point, CHOICE_POINT_VALUE],
        )
        events, trail_size = tighten_at(state, trail_log, trail_indices, mark, trail_top[0], variable, new_min, new_max)
        trail_top[0] = trail_size
        lit_top[0] = lit_push_events(
            lit_trail, lit_last, lit_top[0], state, variable, events, choice_point, REASON_DECISION, -1, -1
        )
        if events != EVENT_MASK_NONE:
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                priorities,
                variable,
                events,
                -1,
            )
        if optimized_variable < 0:
            return True
        events = tighten_objective(
            state,
            trail_log,
            trail_top,
            trail_indices,
            mark,
            optimized_variable,
            bb[BB_VALUE],
            bb[BB_BOUND],
        )
        if events < 0:  # the choice point cannot hold an improving solution, keep popping
            continue
        if events != EVENT_MASK_NONE:
            lit_top[0] = lit_push_events(
                lit_trail, lit_last, lit_top[0], state, optimized_variable, events, choice_point, REASON_ROOT, -1, -1
            )
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                priorities,
                optimized_variable,
                events,
                -1,
            )
        return True
    return False
