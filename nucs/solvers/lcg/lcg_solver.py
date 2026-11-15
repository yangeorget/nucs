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

from nucs.constants import BB_BOUND, BB_VALUE, BB_VARIABLE, DOMAIN_MAX, DOMAIN_MIN, EVENT_MASK_NONE, LOG_LEVEL_INFO
from nucs.numba_helper import DomainHeuristicFunctions, VariableHeuristicFunctions, addresses_from_functions
from nucs.problems.problem import INFO_ALGORITHM, PROBLEM_BOUND, PROBLEM_INCONSISTENT, PROBLEM_UNBOUND, Problem
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
from nucs.solvers.lcg.clause_store import (
    STORE_CLAUSE_NB,
    STORE_LIT_NB,
    STORE_UNIT_NB,
    allocate_clause_store,
    apply_units,
    attach_clause,
    store_clause,
    store_has_room,
    store_unit,
)
from nucs.solvers.lcg.conflict_analysis import (
    CLAUSE_LOG_ASSERTION_LEVEL,
    CLAUSE_LOG_LEVEL,
    CLAUSE_LOG_LIT_NB,
    CLAUSE_LOG_SOLUTION_NB,
    CLAUSE_LOG_START,
    CLAUSE_LOG_TOP_CLAUSE_NB,
    CLAUSE_LOG_TOP_LIT_NB,
    allocate_analysis,
    allocate_clause_log,
    analyze_conflict,
    clause_log_add,
)
from nucs.solvers.lcg.decisions import (
    PENDING_LIT_VAR_SIDE,
    decide,
    decision_literal,
    take_pending_lit,
)
from nucs.solvers.lcg.explanations import (
    EXPLANATION_NOT_TRUE,
    EXPLANATION_VERDICTS,
    EXPLANATIONS_VALID,
    check_explanations,
)
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
from nucs.solvers.lcg.lcg_algorithm import (
    FAILED_CLAUSE,
    FAILED_POS,
    FAILED_PROPAGATOR,
    FAILED_WIDTH,
    PROBLEM_ITRAIL_FULL,
    lcg_algorithm,
)
from nucs.solvers.lcg.literals import LIT_VALUE, LIT_VAR_SIDE
from nucs.solvers.propagation_steps import update_propagators
from nucs.solvers.restart_policies import RESTART_POLICY_NEVER, RestartPolicy
from nucs.solvers.search_steps import record_decision, record_failure, select_variable, stop_status
from nucs.solvers.searches import SEARCHES_DEFAULT, Search
from nucs.solvers.state import TIGHTENING_UTRAIL_ENTRY_NB, tighten_at, utrail_undo
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
# handles these five itself, in _run_search_loop: TreeSearchSolver never sees them.
SOLVER_ITRAIL_FULL = 5  # the implication trail needs more room
SOLVER_ITRAIL_INVALID = 6  # in debug mode, the implication trail does not agree with the state
SOLVER_EXPLANATION_INVALID = 7  # in debug mode, an explanation is not proven correct
SOLVER_CLAUSE_LOG_FULL = 8  # in debug mode, the log of the learned clauses needs more room
SOLVER_CLAUSES_FULL = 9  # the store of the learned clauses needs more room


class LcgSolver(TreeSearchSolver):
    """
    A solver that will learn clauses from its failures (lazy clause generation).

    Its propagation is always bound consistency, and no parameter selects another one: the conflict analysis needs a
    reason for each change of a bound, and a custom consistency algorithm gives none.

    At this step (design/proposals/lcg.md, 7.0, step 9), it records each bound change on the path, with its choice
    point and its reason, on the implication trail (nucs.solvers.lcg.implication_trail), and each decision is one
    literal on its own choice point (nucs.solvers.lcg.decisions). At each failure, it learns a clause by the 1UIP
    analysis (nucs.solvers.lcg.conflict_analysis), and adds it to the store, which propagates the clauses
    (nucs.solvers.lcg.clause_store). The search still backtracks in the classical way: it finds the solutions and the
    optima of BacktrackSolver, and the clauses remove nodes from its tree (with an adaptive heuristic, such as
    dom/wdeg, they can also change the order of the decisions). The debug mode keeps the learned
    clauses (learned_clauses), so that the tests can check them.
    Apart from the clauses, it is the search of BacktrackSolver, except that an EQ decision inside the domain is two
    literals on two choice points.
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
                      result with the state, checks the explanation of each change that a propagator makes and of
                      each failure, and keeps the learned clauses; an AssertionError tells the first row that does not
                      agree, or the first explanation that is not proven correct. This is slow: it is for the tests.
                      Defaults to False
        :type debug: bool
        """
        # allocated before TreeSearchSolver.__init__, because it calls _record_root
        self.itrail, self.itrail_top, self.itrail_last = allocate_itrail(problem.domain_nb)
        # the second literal of an interior EQ, while the first one propagates (see nucs.solvers.lcg.decisions)
        self.pending_lit = np.full(2, -1, dtype=np.int32)
        # the propagator that failed last and the position of its call, which lcg_algorithm writes; the conflict is
        # explained with them
        self.failed_propagator = np.full(FAILED_WIDTH, -1, dtype=np.int32)
        # the store of the learned clauses, and the first row whose watches the clause propagation has not visited
        (
            self.clause_lits,
            self.clause_start,
            self.watch_head,
            self.watches,
            self.unit_lits,
            self.store_top,
            self.clause_head,
        ) = allocate_clause_store(problem.domain_nb)
        # 1 when a unit clause is false at the root, which then holds no solution
        self.root_failed = np.zeros(1, dtype=np.int32)
        # in debug mode, the row of the first explanation that is not proven correct, and its verdict
        self.explanation_report = np.full(2, -1, dtype=np.int64)
        self.debug = debug
        super().__init__(problem, searches, log_level, weight_decay, restart_policy, last_conflict)
        # the compiled explain address of each algorithm, resolved as compute_domains_addrs is: only for the
        # algorithms of the problem, and as a placeholder without the JIT
        self.explain_addrs = addresses_from_functions(
            EXPLAIN_FCTS, SIGN_EXPLAIN, np.unique(self.problem.propagator_infos[:, INFO_ALGORITHM]), ALG_DUMMY
        )
        # the buffers of the conflict analysis, which runs at each failure
        self.explanation, self.explanation_lits, self.marks, self.marked, self.learned = allocate_analysis(
            problem.domain_nb, len(self.domain_buffer)
        )
        # the learned clauses, which only the debug mode logs: without it, the log has no rows
        self.clause_log, self.clause_log_info, self.clause_log_top = allocate_clause_log(problem.domain_nb, debug)
        logger.info(f"The implication trail starts at {len(self.itrail)} rows and grows when it runs out")
        logger.info(
            f"LcgSolver checks the implication trail and the explanations at each node: {'yes' if debug else 'no'}"
        )

    def _grow_clause_store(self) -> None:
        """
        Doubles each array of the store of the learned clauses that has no room for one more clause.
        """
        clause_nb = int(self.store_top[STORE_CLAUSE_NB])
        if self.store_top[STORE_LIT_NB] + len(self.learned) > len(self.clause_lits):
            self.clause_lits = double_rows(self.clause_lits)
        if clause_nb + 2 > len(self.clause_start):
            self.clause_start = double_rows(self.clause_start)
        if (clause_nb + 1) << 1 > len(self.watches):
            self.watches = double_rows(self.watches)
        if self.store_top[STORE_UNIT_NB] + 1 > len(self.unit_lits):
            self.unit_lits = double_rows(self.unit_lits)
        logger.info(f"The store of the learned clauses grew, at {clause_nb} clauses")

    def learned_clauses(self) -> list[tuple[list[tuple[int, int]], int, int, int]]:
        """
        Returns the clauses that the conflict analysis learned, in the order of the failures. Only the debug mode
        keeps them: without it, the list is empty.

        :return: for each clause, its literals as (var_side, value) pairs with the asserting literal first, the level of
                 the conflict, the assertion level (-1 when the clause does not assert a literal), and the number of
                 solutions that the search had found when it learned the clause
        :rtype: List[Tuple[List[Tuple[int, int]], int, int, int]]
        """
        clauses = []
        for info in self.clause_log_info[: self.clause_log_top[CLAUSE_LOG_TOP_CLAUSE_NB]].tolist():
            start = info[CLAUSE_LOG_START]
            literals = self.clause_log[start : start + info[CLAUSE_LOG_LIT_NB]].tolist()
            clauses.append(
                (
                    [(var_side, value) for var_side, value in literals],
                    info[CLAUSE_LOG_LEVEL],
                    info[CLAUSE_LOG_ASSERTION_LEVEL],
                    info[CLAUSE_LOG_SOLUTION_NB],
                )
            )
        return clauses

    def _start_search(self) -> None:
        """
        Empties the store of the learned clauses, and the log of the debug mode, before a new search.

        A clause is valid only inside the search that learned it: during an enumeration, it can remove the solutions
        found before it; during an optimization, the solutions that are not better than the best one.
        """
        self.store_top.fill(0)
        self.watch_head.fill(-1)
        self.clause_log_top.fill(0)

    def _record_root(self) -> None:
        """
        Builds the implication trail again for the root that the solver has just set, with no pending literal, and
        applies the unit clauses there, before the propagation of the root (design/proposals/lcg.md, 4.7).
        """
        self.pending_lit[PENDING_LIT_VAR_SIDE] = -1
        itrail_root(self.state, self.problem.initial_domains, self.itrail, self.itrail_top, self.itrail_last)
        self.clause_head[0] = 0
        unit_nb = int(self.store_top[STORE_UNIT_NB])
        while self.itrail_top[0] + unit_nb > len(self.itrail):
            self.itrail = double_rows(self.itrail)
        self.root_failed[0] = not apply_units(
            self.state,
            self.utrail_log,
            self.utrail_top,
            self.utrail_indices,
            0,
            self.unit_lits,
            unit_nb,
            self.itrail,
            self.itrail_top,
            self.itrail_last,
            0,
            self.triggered_propagators,
            self.entailed,
            self.problem.triggers,
            self.problem.triggers_offsets,
            self.problem.propagator_infos,
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
            self.unit_lits,
            self.store_top,
            self.clause_head,
        )

    def _run_search_loop(self) -> tuple[int, NDArray | None]:
        """
        Runs the compiled search loop of LcgSolver until it returns, by forwarding the solver state to it.

        It grows the implication trail (SOLVER_ITRAIL_FULL), the store of the learned clauses (SOLVER_CLAUSES_FULL)
        and, in debug mode, the log of the learned clauses (SOLVER_CLAUSE_LOG_FULL) itself when the loop asks for it,
        and calls the loop again.

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
                self.explanation,
                self.explanation_lits,
                self.marks,
                self.marked,
                self.learned,
                self.clause_log,
                self.clause_log_info,
                self.clause_log_top,
                self.clause_lits,
                self.clause_start,
                self.watch_head,
                self.watches,
                self.unit_lits,
                self.store_top,
                self.clause_head,
                self.root_failed,
            )
            if status == SOLVER_ITRAIL_FULL:
                self.itrail = double_rows(self.itrail)
                logger.info(f"The implication trail grew to {len(self.itrail)} rows")
            elif status == SOLVER_CLAUSES_FULL:
                self._grow_clause_store()
            elif status == SOLVER_CLAUSE_LOG_FULL:
                # grow the part that is full; a clause has at most one literal for each bound
                if self.clause_log_top[CLAUSE_LOG_TOP_CLAUSE_NB] == len(self.clause_log_info):
                    self.clause_log_info = double_rows(self.clause_log_info)
                if self.clause_log_top[CLAUSE_LOG_TOP_LIT_NB] + len(self.learned) > len(self.clause_log):
                    self.clause_log = double_rows(self.clause_log)
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
    explanation: NDArray,
    explanation_lits: NDArray,
    marks: NDArray,
    marked: NDArray,
    learned: NDArray,
    clause_log: NDArray,
    clause_log_info: NDArray,
    clause_log_top: NDArray,
    clause_lits: NDArray,
    clause_start: NDArray,
    watch_head: NDArray,
    watches: NDArray,
    unit_lits: NDArray,
    store_top: NDArray,
    clause_head: NDArray,
    root_failed: NDArray,
) -> tuple[int, NDArray | None]:
    """
    Searches for one solution, as the run_search_loop of BacktrackSolver does: the search loop of LcgSolver.

    It is a copy of the run_search_loop of BacktrackSolver with five changes (design/proposals/lcg.md, 7.0, steps 5,
    6 and 8), and a debug mode (steps 5, 7 and 8):

    - it calls lcg_algorithm directly, where that loop calls the consistency algorithm of the solver; lcg_algorithm
      records the changes that the propagators make on the implication trail;
    - each decision is one literal on one choice point (decide), where branch parks two alternatives for an EQ inside
      the domain: the second literal of such an EQ waits in pending_lit, and comes before the heuristics at the next
      node if the propagation left it open;
    - a decision writes the size of the implication trail on the choice point that it parks
      (CHOICEPOINT_ITRAIL_MARK), and adds its row with REASON_DECISION;
    - it backtracks with the backtrack of LcgSolver, which also pops the implication trail;
    - at each failure of a propagator or of a clause, it learns a clause by the 1UIP analysis (analyze_conflict), and
      adds it to the store (step 9): lcg_algorithm propagates the stored clauses, but the search still backtracks in
      the classical way. After the backtrack, the new clause gets its watches, and sets its literal when it is unit
      (attach_clause); when the resumed choice point violates it, the search backtracks again. A clause with one
      literal goes to unit_lits, which each backtrack and each return to the root apply; the empty clause ends the
      search. An analysis that meets a wrong explanation gives no clause; the debug mode stops on it, as on an
      explanation that is not proven correct.

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
    :param explanation: a scratch buffer of the conflict analysis, for the explanations
    :type explanation: NDArray
    :param explanation_lits: a scratch buffer of the conflict analysis, for the literals of the explanations
    :type explanation_lits: NDArray
    :param marks: the marks of the conflict analysis, see MARK_*
    :type marks: NDArray
    :param marked: a scratch buffer of the conflict analysis, for the stack of the marked bounds
    :type marked: NDArray
    :param learned: a buffer that gets the clause that the conflict analysis learns
    :type learned: NDArray
    :param clause_log: the literals of the learned clauses that the debug mode logs
    :type clause_log: NDArray
    :param clause_log_info: one row for each logged clause, see CLAUSE_LOG_*
    :type clause_log_info: NDArray
    :param clause_log_top: the number of clauses and the number of literals in the log, see CLAUSE_LOG_TOP_*
    :type clause_log_top: NDArray
    :param clause_lits: the literals of the learned clauses, see nucs.solvers.lcg.clause_store
    :type clause_lits: NDArray
    :param clause_start: the first literal of each learned clause
    :type clause_start: NDArray
    :param watch_head: the first watch of each bound, -1 when the list is empty
    :type watch_head: NDArray
    :param watches: the watches of the learned clauses
    :type watches: NDArray
    :param unit_lits: the learned clauses with one literal
    :type unit_lits: NDArray
    :param store_top: the numbers of clauses, of literals and of unit clauses of the store, see STORE_*
    :type store_top: NDArray
    :param clause_head: the first row of the implication trail whose watches the clause propagation has not visited
                        yet, as a one-cell array
    :type clause_head: NDArray
    :param root_failed: 1 when a unit clause is false at the root, as a one-cell array
    :type root_failed: NDArray

    :return: why the loop returned, and the solution when it found one
    :rtype: Tuple[int, Optional[NDArray]]
    """
    max_choicepoint = len(choicepoints) - 2  # a decision pushes one choice point and marks a second
    if root_failed[0]:
        return SOLVER_RUNNING, None  # a unit clause is false at the root: no solution is left
    while True:
        # a backtrack also applies the asserting literal of a new clause and each unit clause, at its own mark
        headroom = utrail_headroom + (store_top[STORE_UNIT_NB] + 1) * TIGHTENING_UTRAIL_ENTRY_NB
        status = stop_status(
            interruption, search_control, utrail_log, utrail_top, headroom, choicepoint_top, max_choicepoint
        )
        if status != SOLVER_RUNNING:
            return status, None
        if not store_has_room(clause_lits, clause_start, watches, unit_lits, store_top, len(learned)):
            return SOLVER_CLAUSES_FULL, None
        # no test of the room of the implication trail here: lcg_algorithm makes sure, before each propagator call,
        # that the rows of the call and those of the next decision or backtrack fit
        if debug and not is_itrail_valid(
            state, initial_domains, itrail, itrail_top, itrail_last, choicepoints, choicepoint_top
        ):
            return SOLVER_ITRAIL_INVALID, None
        # the room of the log for the clause of a failure of this propagation, before the propagation starts
        if debug and (
            clause_log_top[CLAUSE_LOG_TOP_CLAUSE_NB] == len(clause_log_info)
            or clause_log_top[CLAUSE_LOG_TOP_LIT_NB] + len(learned) > len(clause_log)
        ):
            return SOLVER_CLAUSE_LOG_FULL, None
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
            clause_lits,
            clause_start,
            watch_head,
            watches,
            store_top,
            clause_head,
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
                explanation,
                explanation_lits,
            )
            if row != EXPLANATIONS_VALID:
                explanation_report[0] = row
                explanation_report[1] = verdict
                return SOLVER_EXPLANATION_INVALID, None
        if problem_status == PROBLEM_ITRAIL_FULL:
            return SOLVER_ITRAIL_FULL, None
        # the clause that the failure of this propagation teaches, -1 when there is none or when it has one literal
        new_clause = -1
        if problem_status == PROBLEM_INCONSISTENT:
            # the search backtracks as before (steps 8 and 9 of design/proposals/lcg.md 7.0), and the store propagates
            # the clause from the choice point where the search resumes
            lit_nb, level, assertion_level = analyze_conflict(
                explain_addrs,
                propagator_infos,
                offsets,
                propagator_variables,
                propagator_parameters,
                initial_domains,
                itrail,
                itrail_top,
                itrail_last,
                failed_propagator[FAILED_PROPAGATOR],
                failed_propagator[FAILED_CLAUSE],
                failed_propagator[FAILED_POS],
                clause_lits,
                clause_start,
                domain_buffer,
                explanation,
                explanation_lits,
                marks,
                marked,
                learned,
            )
            if debug:
                if lit_nb < 0:
                    # an explanation gave a literal that was not true when its call read the bounds; level is its row
                    explanation_report[0] = level
                    explanation_report[1] = EXPLANATION_NOT_TRUE
                    return SOLVER_EXPLANATION_INVALID, None
                clause_log_add(
                    clause_log,
                    clause_log_info,
                    clause_log_top,
                    learned,
                    lit_nb,
                    level,
                    assertion_level,
                    statistics[STATS_IDX_SOLUTION_NB],
                )
            if lit_nb == 0:
                return SOLVER_RUNNING, None  # the empty clause: no solution is left
            if lit_nb == 1:
                store_unit(unit_lits, store_top, learned[0, LIT_VAR_SIDE], learned[0, LIT_VALUE])
            elif lit_nb > 1:
                new_clause = store_clause(clause_lits, clause_start, store_top, learned, lit_nb)
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
            while True:
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
                    unit_lits,
                    store_top,
                    clause_head,
                ):
                    return SOLVER_RUNNING, None
                # the new clause gets its watches at the choice point where the search resumes; when all its literals
                # are false there, the choice point holds no solution, and the search backtracks again
                if new_clause < 0 or attach_clause(
                    new_clause,
                    clause_lits,
                    clause_start,
                    watch_head,
                    watches,
                    state,
                    utrail_log,
                    utrail_top,
                    utrail_indices,
                    choicepoints[choicepoint_top[0], CHOICEPOINT_UTRAIL_MARK],
                    itrail,
                    itrail_top,
                    itrail_last,
                    choicepoint_top[0],
                    triggered_propagators,
                    entailed,
                    triggers,
                    triggers_offsets,
                    propagator_infos,
                ):
                    break
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
    unit_lits: NDArray,
    store_top: NDArray,
    clause_head: NDArray,
) -> bool:
    """
    Backtracks to the deepest choice point that can still hold a solution, as the backtrack of BacktrackSolver
    (nucs.solvers.choicepoints) does, and keeps the implication trail in step.

    It is a copy of that backtrack with four changes: it drops the pending literal of an interior EQ, which belongs to
    the choice point that the backtrack leaves; popping a choice point also pops the implication trail down to the mark
    of that choice point, and moves the head of the clause propagation down with it; the parked alternative and the
    objective bound add their rows, at the choice point that the search resumes; and it applies the unit clauses
    there. The alternative gets REASON_DECISION: at this step it is a decision of the search, as in BacktrackSolver.
    The objective bound and the unit clauses get REASON_ROOT: they hold for the rest of the search. A choice point
    where a unit clause is false holds no solution: the backtrack pops it, as one where the objective bound fails.

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
    :param unit_lits: the learned clauses with one literal
    :type unit_lits: NDArray
    :param store_top: the numbers of clauses, of literals and of unit clauses of the store, see STORE_*
    :type store_top: NDArray
    :param clause_head: the first row of the implication trail whose watches the clause propagation has not visited
                        yet, as a one-cell array
    :type clause_head: NDArray

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
        clause_head[0] = min(clause_head[0], itrail_top[0])
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
        if optimized_variable >= 0:
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
        # the unit clauses hold for the rest of the search, as the objective bound does
        if not apply_units(
            state,
            utrail_log,
            utrail_top,
            utrail_indices,
            mark,
            unit_lits,
            store_top[STORE_UNIT_NB],
            itrail,
            itrail_top,
            itrail_last,
            choicepoint,
            triggered_propagators,
            entailed,
            triggers,
            triggers_offsets,
            propagator_infos,
        ):
            continue  # a unit clause is false at the choice point, keep popping
        return True
    return False
