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

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.constants import (
    BB_BOUND,
    BB_VALUE,
    BB_VARIABLE,
    DOMAIN_MAX,
    DOMAIN_MIN,
    EVENT_MASK_NONE,
    INFO_ALGORITHM,
    LOG_LEVEL_INFO,
)
from nucs.numba_helper import DomainHeuristicFunctions, VariableHeuristicFunctions, addresses_from_functions
from nucs.problems.problem import PROBLEM_BOUND, PROBLEM_INCONSISTENT, PROBLEM_UNBOUND, Problem
from nucs.propagators.propagators import ALG_DUMMY, EXPLAIN_FCTS, SIGN_EXPLAIN
from nucs.solvers.choicepoints import (
    CHOICEPOINT_BOUND,
    CHOICEPOINT_ITRAIL_MARK,
    CHOICEPOINT_UTRAIL_MARK,
    CHOICEPOINT_VALUE,
    CHOICEPOINT_VARIABLE,
    bounds_after_tightening,
    tighten_objective,
)
from nucs.solvers.lcg.decisions import (
    PENDING_LIT_VAR_SIDE,
    decide,
    decision_literal,
    take_pending_lit,
)
from nucs.solvers.lcg.explanations import EXPLANATION_VERDICTS, EXPLANATIONS_VALID, check_explanations
from nucs.solvers.lcg.implication_trail import (
    ITRAIL_VALID,
    REASON_DECISION,
    REASON_ROOT,
    allocate_itrail,
    itrail_check,
    itrail_pop,
    itrail_push_events,
    itrail_root,
)
from nucs.solvers.lcg.lcg_algorithm import FAILED_POS, FAILED_PROPAGATOR, PROBLEM_ITRAIL_FULL, lcg_algorithm
from nucs.solvers.propagation_steps import update_propagators
from nucs.solvers.restart_policies import RESTART_POLICY_NEVER, RestartPolicy
from nucs.solvers.search_steps import record_decision, record_failure, select_variable, stop_status
from nucs.solvers.searches import SEARCHES_DEFAULT, Search
from nucs.solvers.state import tighten_at, utrail_undo
from nucs.solvers.tree_search_solver import (
    SOLVER_RUNNING,
    TreeSearchSolver,
    double_rows,
    get_solution,
)
from nucs.statistics import (
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
)

logger = logging.getLogger(__name__)

# Why the search loop of LcgSolver returned, in addition to the SOLVER_* statuses of tree_search_solver.py. LcgSolver
# handles these three itself, in _run_search_loop: TreeSearchSolver never sees them.
SOLVER_ITRAIL_FULL = 5  # the implication trail needs more room
SOLVER_ITRAIL_INVALID = 6  # in debug mode, the implication trail does not agree with the state
SOLVER_EXPLANATION_INVALID = 7  # in debug mode, an explanation is not proven correct


class LcgSolver(TreeSearchSolver):
    """
    A solver that will learn clauses from its failures (lazy clause generation).

    Its propagation is always bound consistency, and no parameter selects another one: the conflict analysis needs a
    reason for each change of a bound, and a custom consistency algorithm gives none.

    At this step (design/proposals/lcg.md, 7.0, step 7), it records each bound change on the path, with its choice
    point and its reason, on the implication trail (nucs.solvers.lcg.implication_trail), and each decision is one
    literal on its own choice point (nucs.solvers.lcg.decisions). It can explain each change that a propagator made,
    and each failure (nucs.solvers.lcg.explanations), but only the debug mode asks for the explanations. The search
    does not use the implication trail yet.
    It is the search of BacktrackSolver, except that an EQ decision inside the domain is two literals on two choice
    points: it finds the same solutions and the same optima, and when no decision is such an EQ, every counter is the
    same.
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
                      result with the state, and checks the explanation of each change that a propagator makes and of
                      each failure; an AssertionError tells the first row that does not agree, or the first explanation
                      that is not proven correct. This is slow: it is for the tests. Defaults to False
        :type debug: bool
        """
        # allocated before TreeSearchSolver.__init__, because it calls _record_root
        self.itrail, self.itrail_top, self.itrail_last = allocate_itrail(problem.domain_nb)
        # the second literal of an interior EQ, while the first one propagates (see nucs.solvers.lcg.decisions)
        self.pending_lit = np.full(2, -1, dtype=np.int32)
        # the propagator that failed last and the position of its call, which lcg_algorithm writes; the conflict is
        # explained with them
        self.failed_propagator = np.full(2, -1, dtype=np.int32)
        # in debug mode, the row of the first explanation that is not proven correct, and its verdict
        self.explanation_report = np.full(2, -1, dtype=np.int64)
        self.debug = debug
        super().__init__(problem, searches, log_level, weight_decay, restart_policy, last_conflict)
        # the compiled explain address of each algorithm, resolved as compute_domains_addrs is: only for the
        # algorithms of the problem, and as a placeholder without the JIT
        self.explain_addrs = addresses_from_functions(
            EXPLAIN_FCTS, SIGN_EXPLAIN, np.unique(self.problem.propagator_infos[:, INFO_ALGORITHM]), ALG_DUMMY
        )
        logger.info(f"The implication trail starts at {len(self.itrail)} rows and grows when it runs out")
        logger.info(
            f"LcgSolver checks the implication trail and the explanations at each node: {'yes' if debug else 'no'}"
        )

    def _record_root(self) -> None:
        """
        Builds the implication trail again for the root that the solver has just set, with no pending literal.
        """
        self.pending_lit[PENDING_LIT_VAR_SIDE] = -1
        itrail_root(self.state, self.problem.initial_domains, self.itrail, self.itrail_top, self.itrail_last)

    def _backtrack(self) -> bool:
        """
        Backtracks by forwarding the solver state to the jitted backtrack of LcgSolver.

        :return: true iff it was possible to backtrack
        :rtype: bool
        """
        return backtrack(
            self.statistics,
            self.state,
            self.utrail_log,
            self.utrail_top,
            self.utrail_indices,
            self.choicepoints,
            self.choicepoint_top,
            self.entailed,
            self.triggered_propagators,
            self.problem.triggers,
            self.problem.triggers_offsets,
            self.problem.propagator_infos,
            self.bb,
            self.itrail,
            self.itrail_top,
            self.itrail_last,
            self.pending_lit,
        )

    def _run_search_loop(self) -> tuple[int, NDArray | None]:
        """
        Runs the compiled search loop of LcgSolver until it returns, by forwarding the solver state to it.

        It grows the implication trail itself when the loop asks for it, and calls the loop again.

        :return: why the loop returned, and the solution when it found one
        :rtype: Tuple[int, Optional[NDArray]]
        """
        while True:
            status, solution = run_search_loop(
                self.statistics,
                self.problem.propagator_infos,
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
                self.choicepoints,
                self.choicepoint_top,
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
                self.utrail_headroom,
                self.interrupter.cells,
                self.propagator_weights,
                self.search_control,
                self.flat_searches.variable_searches,
                self.itrail,
                self.itrail_top,
                self.itrail_last,
                self.pending_lit,
                self.problem.initial_domains,
                self.debug,
                self.explain_addrs,
                self.failed_propagator,
                self.explanation_report,
            )
            if status == SOLVER_ITRAIL_FULL:
                self.itrail = double_rows(self.itrail)
                logger.info(f"The implication trail grew to {len(self.itrail)} rows")
            elif status == SOLVER_ITRAIL_INVALID:
                row = itrail_check(
                    self.state,
                    self.problem.initial_domains,
                    self.itrail,
                    self.itrail_top,
                    self.itrail_last,
                    self.choicepoints,
                    self.choicepoint_top[0],
                )
                raise AssertionError(
                    f"The implication trail does not agree with the state at row {row} of {self.itrail_top[0]}"
                )
            elif status == SOLVER_EXPLANATION_INVALID:
                row, verdict = (int(cell) for cell in self.explanation_report)
                what = (
                    f"the failure of propagator {self.failed_propagator[FAILED_PROPAGATOR]}"
                    if row == self.itrail_top[0]
                    else f"row {row} of {self.itrail_top[0]}"
                )
                raise AssertionError(f"The explanation of {what} is {EXPLANATION_VERDICTS[verdict]}")
            else:
                return status, solution


@njit(cache=True, nogil=True)
def run_search_loop(
    statistics: NDArray,
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
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    pending_lit: NDArray,
    initial_domains: NDArray,
    debug: bool,
    explain_addrs: NDArray,
    failed_propagator: NDArray,
    explanation_report: NDArray,
) -> tuple[int, NDArray | None]:
    """
    Searches for one solution, as the run_search_loop of BacktrackSolver does: the search loop of LcgSolver.

    It is a copy of the run_search_loop of BacktrackSolver with four changes (design/proposals/lcg.md, 7.0, steps 5
    and 6), and a debug mode (steps 5 and 7):

    - it calls lcg_algorithm directly, where that loop calls the consistency algorithm of the solver; lcg_algorithm
      records the changes that the propagators make on the implication trail;
    - each decision is one literal on one choice point (decide), where branch parks two alternatives for an EQ inside
      the domain: the second literal of such an EQ waits in pending_lit, and comes before the heuristics at the next
      node if the propagation left it open;
    - a decision writes the size of the implication trail on the choice point that it parks
      (CHOICEPOINT_ITRAIL_MARK), and adds its row with REASON_DECISION;
    - it backtracks with the backtrack of LcgSolver, which also pops the implication trail.

    The choice point of a row is choicepoint_top[0] when the row is written. STATS_IDX_SOLVER_CHOICE_NB counts the
    decision literals, so an interior EQ counts two choices when its second literal opens a choice point.

    It stops early when an array it cannot grow runs out of room.

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
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray
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
    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray
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
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param pending_lit: the second literal of an interior EQ, see PENDING_LIT_*
    :type pending_lit: NDArray
    :param initial_domains: the domains of the problem before any propagation, for the debug check
    :type initial_domains: NDArray
    :param debug: whether to replay the implication trail and compare it with the state at each node, and to check
                  the explanation of each row that the propagation writes and of each failure
    :type debug: bool
    :param explain_addrs: the compiled explain address of each algorithm, for the debug mode
    :type explain_addrs: NDArray
    :param failed_propagator: a two-cell array that gets the propagator that failed and the position of its call,
                              from lcg_algorithm
    :type failed_propagator: NDArray
    :param explanation_report: a two-cell array that gets, in debug mode, the row of the first explanation that is
                               not proven correct (the number of rows in use for a failure) and its verdict
    :type explanation_report: NDArray

    :return: why the loop returned, and the solution when it found one
    :rtype: Tuple[int, Optional[NDArray]]
    """
    max_choicepoint = len(choicepoints) - 2  # a decision pushes one choice point and marks a second
    while True:
        status = stop_status(
            interruption, search_control, utrail_log, utrail_top, utrail_headroom, choicepoint_top, max_choicepoint
        )
        if status != SOLVER_RUNNING:
            return status, None
        # no test of the room of the implication trail here: lcg_algorithm makes sure, before each propagator call,
        # that the rows of the call and those of the next decision or backtrack fit
        if debug and not is_itrail_valid(
            state, initial_domains, itrail, itrail_top, itrail_last, choicepoints, choicepoint_top
        ):
            return SOLVER_ITRAIL_INVALID, None
        first_row = itrail_top[0]
        problem_status = lcg_algorithm(
            statistics,
            propagator_weights,
            algorithm_flags,
            propagator_infos,
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
            choicepoints,
            choicepoint_top,
            triggered_propagators,
            compute_domains_addrs,
            domain_buffer,
            itrail,
            itrail_top,
            itrail_last,
            failed_propagator,
        )
        # the rows of a propagation that stops for a full implication trail are checked before the return: the next
        # call checks only its own rows
        if debug:
            row, verdict = check_explanations(
                compute_domains_addrs,
                explain_addrs,
                propagator_infos,
                offsets,
                propagator_variables,
                propagator_parameters,
                initial_domains,
                itrail,
                itrail_top,
                itrail_last,
                first_row,
                failed_propagator[FAILED_PROPAGATOR] if problem_status == PROBLEM_INCONSISTENT else -1,
                failed_propagator[FAILED_POS],
                domain_buffer,
            )
            if row != EXPLANATIONS_VALID:
                explanation_report[0] = row
                explanation_report[1] = verdict
                return SOLVER_EXPLANATION_INVALID, None
        if problem_status == PROBLEM_ITRAIL_FULL:
            return SOLVER_ITRAIL_FULL, None
        if debug and not is_itrail_valid(
            state, initial_domains, itrail, itrail_top, itrail_last, choicepoints, choicepoint_top
        ):
            return SOLVER_ITRAIL_INVALID, None
        if problem_status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLUTION_NB] += 1
            return SOLVER_RUNNING, get_solution(domains)
        # the variable to branch on, -1 when there is none, and the search that owns it. var_side is the bound of the
        # decision literal when it is the pending literal of an interior EQ, -1 when the heuristics choose the decision.
        search_idx = 0
        variable = -1
        var_side = -1
        value = 0
        if problem_status == PROBLEM_UNBOUND:
            var_side, value = take_pending_lit(state, pending_lit)
            if var_side >= 0:
                variable = var_side >> 1
            else:
                search_idx, variable = select_variable(
                    domains,
                    entailed,
                    offsets,
                    propagator_variables,
                    triggers,
                    triggers_offsets,
                    propagator_weights,
                    search_control,
                    variable_searches,
                    decision_variables,
                    decision_variables_offsets,
                    var_heuristic_fcts,
                    var_heuristic_params,
                    var_heuristic_params_offsets,
                    var_heuristic_params_shapes,
                )
        # either the problem is inconsistent, or no search can claim a variable although variables remain
        # unbound -- a choice point whose domains admit no assignment. Both are dead ends: backtrack.
        if variable == -1:
            record_failure(search_control)
            if not backtrack(
                statistics,
                state,
                utrail_log,
                utrail_top,
                utrail_indices,
                choicepoints,
                choicepoint_top,
                entailed,
                triggered_propagators,
                triggers,
                triggers_offsets,
                propagator_infos,
                bb,
                itrail,
                itrail_top,
                itrail_last,
                pending_lit,
            ):
                return SOLVER_RUNNING, None
            continue
        if var_side < 0:
            # the heuristic only says where to split; decision_literal maps the split to the literal to apply now, and
            # puts the second literal of an interior EQ in pending_lit
            kind, split_value = dom_heuristic_fcts[search_idx](
                domains,
                variable,
                dom_heuristic_params[
                    dom_heuristic_params_offsets[search_idx] : dom_heuristic_params_offsets[search_idx + 1]
                ].reshape(dom_heuristic_params_shapes[search_idx, 0], dom_heuristic_params_shapes[search_idx, 1]),
            )
            var_side, value = decision_literal(
                domains[variable, DOMAIN_MIN], domains[variable, DOMAIN_MAX], variable, kind, split_value, pending_lit
            )
        parked = choicepoint_top[0]
        events = decide(state, utrail_log, utrail_top, utrail_indices, choicepoints, choicepoint_top, var_side, value)
        choicepoint = choicepoint_top[0]
        # the parked choice point branched at this row of the implication trail; the row of the decision is at the new
        # choice point
        itrail_sz = itrail_top[0]
        choicepoints[parked, CHOICEPOINT_ITRAIL_MARK] = itrail_sz
        itrail_top[0] = itrail_push_events(
            itrail, itrail_last, itrail_sz, state, variable, events, choicepoint, REASON_DECISION, -1, -1, -1
        )
        update_propagators(
            triggered_propagators, entailed, triggers, triggers_offsets, propagator_infos, variable, events, -1
        )
        record_decision(statistics, search_control, variable, choicepoint)


@njit(cache=True, inline="always")
def is_itrail_valid(
    state: NDArray,
    initial_domains: NDArray,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
) -> bool:
    """
    Returns whether the implication trail agrees with the state, at the current choice point: the debug check.

    :param state: all the backtrackable state
    :type state: NDArray
    :param initial_domains: the domains of the problem before any propagation
    :type initial_domains: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param choicepoints: the per-choice-point metadata
    :type choicepoints: NDArray
    :param choicepoint_top: the index of the top of the choice points as a Numpy array
    :type choicepoint_top: NDArray

    :return: true iff the implication trail agrees with the state
    :rtype: bool
    """
    return (
        itrail_check(state, initial_domains, itrail, itrail_top, itrail_last, choicepoints, choicepoint_top[0])
        == ITRAIL_VALID
    )


@njit(cache=True)
def backtrack(
    statistics: NDArray,
    state: NDArray,
    utrail_log: NDArray,
    utrail_top: NDArray,
    utrail_indices: NDArray,
    choicepoints: NDArray,
    choicepoint_top: NDArray,
    entailed: NDArray,
    triggered_propagators: NDArray,
    triggers: NDArray,
    triggers_offsets: NDArray,
    propagator_infos: NDArray,
    bb: NDArray,
    itrail: NDArray,
    itrail_top: NDArray,
    itrail_last: NDArray,
    pending_lit: NDArray,
) -> bool:
    """
    Backtracks to the deepest choice point that can still hold a solution, as the backtrack of BacktrackSolver
    (nucs.solvers.choicepoints) does, and keeps the implication trail in step.

    It is a copy of that backtrack with three changes: it drops the pending literal of an interior EQ, which belongs to
    the choice point that the backtrack leaves; popping a choice point also pops the implication trail down to the mark
    of that choice point; and the parked alternative and the objective bound add their rows, at the choice point that
    the search resumes. The alternative gets REASON_DECISION: at this step it is a decision of the search, as in
    BacktrackSolver. The objective bound gets REASON_ROOT: it holds for the rest of the search.

    :param statistics: the statistics array
    :type statistics: NDArray
    :param state: all the backtrackable state
    :type state: NDArray
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
    :param entailed: whether each propagator is entailed, a view of state
    :type entailed: NDArray
    :param triggered_propagators: the set propagators that are currently triggered as a Numpy array
    :type triggered_propagators: NDArray
    :param triggers: a Numpy array of event masks indexed by variables and propagators
    :type triggers: NDArray
    :param triggers_offsets: the CSR offsets delimiting each (variable, event) slice of triggers
    :type triggers_offsets: NDArray
    :param propagator_infos: the algorithm, the queue bucket and the copies flag of each propagator, in the INFO_*
                             columns
    :type propagator_infos: NDArray
    :param bb: the branch-and-bound bound as a Numpy array of variable, bound and value,
               whose variable is -1 when not optimizing
    :type bb: NDArray
    :param itrail: the implication trail
    :type itrail: NDArray
    :param itrail_top: the number of rows in use, as a one-cell array
    :type itrail_top: NDArray
    :param itrail_last: the last row of each bound, -1 when the bound has its root value
    :type itrail_last: NDArray
    :param pending_lit: the second literal of an interior EQ, see PENDING_LIT_*
    :type pending_lit: NDArray

    :return: true iff it is possible to backtrack
    :rtype: bool
    """
    pending_lit[PENDING_LIT_VAR_SIDE] = -1
    optimized_variable = bb[BB_VARIABLE]
    while choicepoint_top[0] > 0:
        choicepoint_top[0] -= 1
        choicepoint = choicepoint_top[0]
        statistics[STATS_IDX_SOLVER_BACKTRACK_NB] += 1
        mark = choicepoints[choicepoint, CHOICEPOINT_UTRAIL_MARK]
        utrail_undo(state, utrail_log, utrail_indices, utrail_top, mark)
        # the rows that this choice point wrote before it branched stay: the rows of the choice points above it go
        itrail_pop(itrail, itrail_top, itrail_last, choicepoints[choicepoint, CHOICEPOINT_ITRAIL_MARK])
        variable = choicepoints[choicepoint, CHOICEPOINT_VARIABLE]
        cell_idx = variable << 1
        new_min, new_max = bounds_after_tightening(
            state[cell_idx],
            state[cell_idx | 1],
            choicepoints[choicepoint, CHOICEPOINT_BOUND],
            choicepoints[choicepoint, CHOICEPOINT_VALUE],
        )
        events, utrail_sz = tighten_at(
            state, utrail_log, utrail_indices, mark, utrail_top[0], variable, new_min, new_max
        )
        utrail_top[0] = utrail_sz
        itrail_top[0] = itrail_push_events(
            itrail, itrail_last, itrail_top[0], state, variable, events, choicepoint, REASON_DECISION, -1, -1, -1
        )
        if events != EVENT_MASK_NONE:
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                propagator_infos,
                variable,
                events,
                -1,
            )
        if optimized_variable < 0:
            return True
        events = tighten_objective(
            state,
            utrail_log,
            utrail_top,
            utrail_indices,
            mark,
            optimized_variable,
            bb[BB_VALUE],
            bb[BB_BOUND],
        )
        if events < 0:  # the choice point cannot hold an improving solution, keep popping
            continue
        if events != EVENT_MASK_NONE:
            itrail_top[0] = itrail_push_events(
                itrail,
                itrail_last,
                itrail_top[0],
                state,
                optimized_variable,
                events,
                choicepoint,
                REASON_ROOT,
                -1,
                -1,
                -1,
            )
            update_propagators(
                triggered_propagators,
                entailed,
                triggers,
                triggers_offsets,
                propagator_infos,
                optimized_variable,
                events,
                -1,
            )
        return True
    return False
