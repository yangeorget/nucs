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
import logging
import time
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from nucs.constants import (
    BB_BOUND,
    BB_VALUE,
    BB_VARIABLE,
    BB_WIDTH,
    DOMAIN_MAX,
    DOMAIN_MIN,
    LOG_LEVEL_INFO,
)
from nucs.heuristics.heuristics import DOM_HEURISTIC_FCTS, SIGN_DOM_HEURISTIC, SIGN_VAR_HEURISTIC, VAR_HEURISTIC_FCTS
from nucs.numba_helper import (
    addresses_from_functions,
    build_function_ptrs,
)
from nucs.problems.problem import OFFSETS_VARIABLE, Problem
from nucs.propagators.propagators import (
    ALG_DUMMY,
    COMPUTE_DOMAINS_FCTS,
    SIGN_COMPUTE_DOMAINS,
    get_algorithm_names,
    get_algorithm_nb,
)
from nucs.solvers.buckets import buckets_create, buckets_init
from nucs.solvers.choice_points import CHOICE_POINT_WIDTH, backtrack, choice_point_init, tighten_objective
from nucs.solvers.interrupter import INTERRUPTION_EXTERNAL, Interrupter
from nucs.solvers.restart_policies import RESTART_POLICY_NEVER, RestartPolicy
from nucs.solvers.search_loops import (
    SEARCH_CONTROL_CONFLICT_VARIABLE,
    SEARCH_CONTROL_DECISION_VARIABLE,
    SEARCH_CONTROL_FAILURE_NB,
    SEARCH_CONTROL_LAST_CONFLICT,
    SEARCH_CONTROL_RESTART_LIMIT,
    SEARCH_CONTROL_WIDTH,
    SOLVER_CHOICE_POINTS_FULL,
    SOLVER_INTERRUPTED,
    SOLVER_RESTART,
    SOLVER_RUNNING,
    SOLVER_TRAIL_FULL,
)
from nucs.solvers.searches import SEARCHES_DEFAULT, Search, flatten_searches
from nucs.solvers.solver import OPTIM_RESET, Solver, format_timeout
from nucs.solvers.state import TIGHTENING_TRAIL_ENTRY_NB
from nucs.solvers.weights import weights_init
from nucs.statistics import (
    STATS_IDX_SOLVER_ELAPSED_TIME,
    STATS_IDX_SOLVER_RESTART_NB,
    statistics_as_dictionary,
    statistics_init,
)

logger = logging.getLogger(__name__)

# Trail entries a step of the search needs beyond one per cell of the backtrackable state.
# The barrier in trail_set trails each cell at most once per choice point, so a fixpoint cannot need more
# than len(state) entries however long it runs. The tightenings the search applies around it are not
# covered by that budget: each writes at a mark the trail holds nothing for yet, so every one of their
# writes is trailed. A tightening writes at most TIGHTENING_TRAIL_ENTRY_NB entries (see nucs.solvers.state);
# a step applies at most two of them -- branch's decision is one, while backtracking a choice point applies
# its parked alternative and then the branch-and-bound objective bound.
STEP_TIGHTENING_NB = 2  # an alternative then the objective bound, the longer of the two ways out of a step


class TreeSearchSolver(Solver):
    """
    The base class of the solvers that explore a search tree: BacktrackSolver and LcgSolver.

    It holds what the two solvers share: the state, the trail, the stack of choice points, the searches, the restarts,
    the optimization modes, the failure weights and the time accounting. A subclass gives only the compiled search
    loop, through _run_search_loop: this is where the two solvers differ.
    """

    def __init__(
        self,
        problem: Problem,
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
        super().__init__(problem, log_level)
        # every search keeps its own decision variables, variable and domain heuristics and their parameters
        self.flat_searches = flatten_searches(searches, problem.domain_nb)
        for description in self.flat_searches.searches_as_printable():
            logger.info(f"{type(self).__name__} uses {description}")
        logger.info(f"{type(self).__name__} uses {restart_policy}")
        logger.info(f"{type(self).__name__} uses last-conflict: {'yes' if last_conflict else 'no'}")
        logger.info(f"{type(self).__name__} uses a weight decay of {weight_decay}")
        self.triggered_propagators = buckets_create(problem.propagator_nb)
        self.domain_buffer = allocate_domain_buffer(problem.offsets)
        (
            self.state,
            self.domains,
            self.entailed,
            self.trail_headroom,
            self.trail_log,
            self.trail_top,
            self.trail_indices,
            self.choice_point_stk,
            self.choice_point_top,
        ) = allocate_search_arrays(problem)
        logger.info(
            f"The stack of choice points starts at {len(self.choice_point_stk)} rows and grows when it runs out"
        )
        logger.info(f"The trail starts at {len(self.trail_log)} entries and grows when it runs out")
        # the branch-and-bound bound, as BB_VARIABLE, BB_BOUND and BB_VALUE: the variable optimized, the side of
        # its domain to tighten, and the best value found so far. It is solver state, not choice-point state --
        # the bound holds for the whole remaining search, so backtrack re-applies it to each choice point it
        # resumes rather than it being written into them all when it is found, and nothing about it is trailed.
        # A return to the root applies it there too, with a mark of 0.
        # BB_VARIABLE stays -1 until the first solution of an optimization, which is how backtrack and
        # _tighten_objective_at_root know there is no bound to apply. In OPTIM_RESET, the root already holds the
        # bound, so backtrack finds it applied and changes nothing.
        self.bb = np.full(BB_WIDTH, -1, dtype=np.int32)
        # read by the search loop at every node, and written by interrupt() and by the deadline of a search
        self.interrupter = Interrupter()
        self.restart_policy = restart_policy
        # the restart limit is drawn by each search, the other cells are reset with the root by _reset_to_root
        self.search_control = np.full(SEARCH_CONTROL_WIDTH, -1, dtype=np.int64)
        self.search_control[SEARCH_CONTROL_LAST_CONFLICT] = last_conflict
        logger.debug("Initializing choice points")
        self._reset_to_root()
        logger.debug("Choice points initialized")
        logger.debug("Initializing statistics")
        self.statistics = statistics_init(get_algorithm_nb())
        logger.debug("Statistics initialized")
        # the failure weight of each propagator: global, never trailed and never reset, so that what one solve
        # learns about the constraints serves the next one too
        self.propagator_weights = weights_init(problem.propagator_nb, weight_decay)
        # the compiled compute_domains address of each algorithm, which the propagation loop calls a propagator
        # through; resolving only the algorithms used by the problem keeps the init cost proportional to the problem
        # instead of the whole propagator library; without the JIT this is a placeholder that call_compute_domains
        # ignores
        self.compute_domains_addrs = addresses_from_functions(
            COMPUTE_DOMAINS_FCTS, SIGN_COMPUTE_DOMAINS, np.unique(self.problem.algorithms), ALG_DUMMY
        )
        # the function tables threaded into the search loop: Numba typed lists under the JIT, plain Python lists
        # otherwise (see build_function_ptrs)
        self.var_heuristic_fcts = build_function_ptrs(
            [VAR_HEURISTIC_FCTS[h] for h in self.flat_searches.var_heuristics], SIGN_VAR_HEURISTIC
        )
        self.dom_heuristic_fcts = build_function_ptrs(
            [DOM_HEURISTIC_FCTS[h] for h in self.flat_searches.dom_heuristics], SIGN_DOM_HEURISTIC
        )
        logger.debug(f"{type(self).__name__} initialized")

    def solve(self, timeout: float | None = None) -> Iterator[NDArray]:
        logger.info(f"Solving and iterating over the solutions, {format_timeout(timeout)}")
        for solution in self._iterate_solutions(self._backtrack_after_solution, timeout):
            logger.debug("Found a solution")
            yield solution

    def _backtrack_after_solution(self, _solution: NDArray) -> bool:
        """
        Moves the enumeration on from a solution, and stops the restarts: a restart would find the solutions
        emitted so far again.

        :param _solution: the solution just found, unused here
        :type _solution: NDArray

        :return: whether the search can continue
        :rtype: bool
        """
        self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = -1
        # the search resumes on a refutation, and a failure that follows one does not name a conflict variable
        self.search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
        return self._backtrack()

    def optimize(self, variable: int, bound: int, mode: str, timeout: float | None = None) -> Iterator[NDArray]:
        logger.info(
            f"Optimizing and iterating over the solutions: {'minimizing' if bound == DOMAIN_MIN else 'maximizing'} "
            f"variable {variable} in {mode} mode, {format_timeout(timeout)}"
        )
        # minimizing a variable means tightening the DOMAIN_MAX side of its domain, and vice versa
        tightened_bound = DOMAIN_MAX if bound == DOMAIN_MIN else DOMAIN_MIN
        for solution in self._iterate_solutions(
            lambda found: self._advance_after_optimum(variable, found[variable], tightened_bound, mode),
            timeout,
        ):
            logger.info(f"Found a local optimum: {solution[variable]}")
            yield solution

    def _iterate_solutions(self, advance: Callable[[NDArray], bool], timeout: float | None) -> Iterator[NDArray]:
        """
        Iterates over the solutions, leaving it to the caller to say how the search moves on from each one.

        That is the only thing enumerating and optimizing do differently: one backtracks to the deepest
        choice point that can still hold a solution, the other tightens the objective and either prunes the
        choice points or restarts from the root. Everything around it is the same search, and getting it
        the same twice is what this avoids -- in particular the elapsed time, which is accounted by
        stopping the clock at each solution and starting it again once the consumer hands control back, so
        that what the consumer does with a solution is not charged to the solver.

        :param advance: called with each solution once the consumer is done with it, returning whether the
                        search can continue
        :type advance: Callable[[NDArray], bool]
        :param timeout: the search budget in seconds, or None for an unbounded search
        :type timeout: Optional[float]

        :return: an iterator over the solutions
        :rtype: Iterator[NDArray]
        """
        self.interrupted = False
        disarm = self.interrupter.arm_deadline(timeout)
        try:
            t0 = time.perf_counter_ns()
            # no solution yet, so the search has no objective bound; _advance_after_optimum arms it.
            # An enumeration disarms it here too: the solver may have been optimized with before.
            self.bb[BB_VARIABLE] = -1
            # the failure limits of the descents: each search draws its own, from the first one, and each restart
            # takes the next one (see _next_solution)
            self.restart_limits = self.restart_policy.limits()
            self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = next(self.restart_limits)
            # a previous search may have left the root tightened by its objective, or wiped out: start from the
            # initial domains, keeping what the propagator weights learned
            self._reset_to_root()
            while (solution := self._next_solution()) is not None:
                self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0
                yield solution
                t0 = time.perf_counter_ns()
                # no check of the budget here: the timer stops the next call of the search loop at its first node,
                # and a search that advance() finds over has nothing left that a timeout could have cut
                if not advance(solution):
                    break
            self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0
        finally:
            # also when the consumer abandons the iteration, so that the timer cannot stop a later search
            disarm()

    def _next_solution(self) -> NDArray | None:
        """
        Searches for the next solution, growing whichever caller-allocated array runs out and resuming.

        :return: the next solution if it exists or None
        :rtype: Optional[NDArray]
        """
        while True:
            status, solution = self._run_search_loop()
            if status == SOLVER_RUNNING:
                return solution
            if status == SOLVER_TRAIL_FULL:
                self.trail_log = double_rows(self.trail_log)
                logger.info(f"The trail grew to {len(self.trail_log)} entries")
            elif status == SOLVER_CHOICE_POINTS_FULL:
                self.choice_point_stk = double_rows(self.choice_point_stk)
                logger.info(f"The stack of choice points grew to a maximal height of {len(self.choice_point_stk)}")
            elif status == SOLVER_RESTART:
                logger.debug("Restarting")
                self.statistics[STATS_IDX_SOLVER_RESTART_NB] += 1
                self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = next(self.restart_limits)
                self._reset_to_root()
                if not self._tighten_objective_at_root():
                    # the root cannot hold a better solution: the best solution so far is proven optimal
                    return None
            elif status == SOLVER_INTERRUPTED:
                # a timeout and interrupt() both cut the search: what it found is not proven
                self.interrupted = True
                if self.interrupter.cells[INTERRUPTION_EXTERNAL]:
                    logger.info("Interrupted, stopping the search")
                else:
                    logger.info("Timeout reached, stopping the search")
                return None
            else:
                raise AssertionError(f"Unknown solver status {status}")

    def _reset_to_root(self) -> None:
        """
        Resets the search to the root: the initial domains, no choice point, every propagator in the queue, and the
        search control that counts from the root -- the failures of the descent and the conflict variable.

        What the search learned stays: the propagator weights are not reset.
        """
        self.search_control[SEARCH_CONTROL_FAILURE_NB] = 0
        self.search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = -1
        self.search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
        choice_point_init(
            self.state,
            self.entailed,
            self.trail_top,
            self.trail_indices,
            self.choice_point_stk,
            self.choice_point_top,
            self.problem.initial_domains,
            self.problem.unbound_variable_nb,
            self.problem.offsets,
        )
        # the search may have stopped at a failure: the failed filtering left propagators in the queue, and backtrack
        # added more, which buckets_init removes before it adds every propagator
        buckets_init(self.triggered_propagators, self.problem.priorities)

    def _tighten_objective_at_root(self) -> bool:
        """
        Applies the bound of the best solution so far at the root, after _reset_to_root, when the objective is armed.

        The objective is tightened with a mark of 0, since the reset undoes every tightening above it: what this
        writes is undone only by the next reset. Its events schedule nobody, since every propagator is in the queue.

        :return: false when the root cannot hold a better solution, which proves the best solution so far optimal
        :rtype: bool
        """
        variable = self.bb[BB_VARIABLE]
        if variable < 0:
            return True
        return (
            tighten_objective(
                self.state,
                self.trail_log,
                self.trail_top,
                self.trail_indices,
                0,
                variable,
                self.bb[BB_VALUE],
                self.bb[BB_BOUND],
            )
            >= 0
        )

    @abstractmethod
    def _run_search_loop(self) -> tuple[int, NDArray | None]:
        """
        Runs the compiled search loop of the solver until it returns, by forwarding the solver state to it.

        :return: why the loop returned, and the solution when it found one
        :rtype: Tuple[int, Optional[NDArray]]
        """
        ...

    def interrupt(self) -> None:
        """
        Stops the search at its next node, as if its budget had run out: the iteration ends and
        :attr:`interrupted` is set. The interruption is final, every later search stops at once too.

        Safe to call from another thread or from a signal handler: it takes no lock, it only writes a cell that
        the compiled search reads, and the search loop releases the GIL so that such a thread gets to run.
        """
        self.interrupter.interrupt()

    def _advance_after_optimum(self, variable: int, value: int, bound: int, mode: str) -> bool:
        """
        After emitting a local optimum, prepares the solver for the next improving solution: either resets to
        the initial domains (OPTIM_RESET) or prunes the choice points, then refixes the objective bound.

        :param variable: the variable being optimized
        :type variable: int
        :param value: the value of the variable in the local optimum just found
        :type value: int
        :param bound: the side of the variable's domain to tighten (DOMAIN_MAX when minimizing, DOMAIN_MIN when maximizing)
        :type bound: int
        :param mode: the optimization mode
        :type mode: str

        :return: whether the search can continue
        :rtype: bool
        """
        # arm the objective in both modes: OPTIM_RESET applies it at the root, and OPTIM_PRUNE lets backtrack
        # apply it to each choice point the search resumes, so the choice points it kills are dropped as they are
        # reached rather than up front
        self.bb[BB_VARIABLE] = variable
        self.bb[BB_BOUND] = bound
        self.bb[BB_VALUE] = value
        if mode == OPTIM_RESET:
            logger.debug("Resetting solver")
            self._reset_to_root()
            return self._tighten_objective_at_root()
        logger.debug("Pruning choice points")
        # the search resumes on a refutation, and a failure that follows one does not name a conflict variable
        self.search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
        return self._backtrack()

    def _backtrack(self) -> bool:
        """
        Backtracks by forwarding the solver state to the jitted backtrack.

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
        )

    def get_statistics_as_dictionary(self) -> dict[str, int]:
        """
        Returns the statistics as a dictionary.

        :return: a dictionary mapping statistic labels to values
        :rtype: Dict[str, int]
        """
        return statistics_as_dictionary(self.statistics, get_algorithm_names())


def allocate_search_arrays(
    problem: Problem,
) -> tuple[NDArray, NDArray, NDArray, int, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Lays out the backtrackable state of a problem and allocates the trail and the stack of choice points.

    :param problem: the problem, already initialized
    :type problem: Problem

    :return: the arrays of the search, in this order: all the backtrackable state; the domains, a (domain_nb, 2) view
             of its head; the entailment flags, a view of its middle; the trail entries that any one step of the
             search can need; the undo log of (cell index, old value) pairs; the trail size, as a one-cell array;
             the index of the last trail entry of each cell, -1 for none; the stack of choice points; the height of
             the stack, as a one-cell array
    :rtype: Tuple[NDArray, NDArray, NDArray, int, NDArray, NDArray, NDArray, NDArray, NDArray]
    """
    # all the backtrackable state in one flat int32 array, so that one undo log and one undo loop
    # restore every kind of it:
    #     [ 2 * domain_nb domain bounds | propagator_nb entailment flags | propagator state | the unbound count ]
    # domains is a (domain_nb, 2) view of its head and entailed a view of the middle -- the same
    # memory, addressed the way each reader wants it -- so the flat index of (variable, bound) is
    # (variable << 1) | bound, and that of propagator p is 2 * domain_nb + p. Restoring a domain
    # bound and reactivating an entailed propagator are then the same instruction. Propagator state
    # (problem.state_width cells, addressed per propagator by offsets[:, OFFSETS_STATE], see
    # Problem.init_propagator_arrays) sits between the entailment flags and the unbound count so that
    # unbound_index() -- len(state) - 1 -- stays independent of it; every state block in this stage
    # defaults its root value to 0 (np.zeros below), which is what every propagator using one expects.
    domain_nb = problem.domain_nb
    propagator_nb = problem.propagator_nb
    propagator_entailment_offset = 2 * domain_nb
    propagator_state_offset = propagator_entailment_offset + propagator_nb
    unbound_count_offset = propagator_state_offset + problem.state_width
    state = np.zeros(unbound_count_offset + 1, dtype=np.int32)
    domains = state[:propagator_entailment_offset].reshape(domain_nb, 2)
    entailed = state[propagator_entailment_offset:propagator_state_offset]
    # the guard lets a choice point trail each *trailable* cell at most once -- every domain bound, every
    # entailment flag, the count, and the trailed prefix of each propagator's state block -- so a fixpoint
    # cannot need more than that many entries, whatever it does; the tightenings the search applies around
    # it write at their own mark and are counted on top. The solver grows the trail when this much room is
    # no longer there. Counted rather than taken as len(state), which would include the untrailed hint
    # suffixes: bc_algorithm only ever trails block p's prefix below offsets[p, OFFSETS_STATE_HINT], so a
    # hint cell can never reach the trail, and a wide alldifferent would otherwise inflate trail_log -- 16x
    # trail_headroom rows of 2 int32 -- by 128 bytes per scratch cell it can never use.
    trailable_nb = propagator_state_offset + problem.state_trailed_width + 1
    trail_headroom = trailable_nb + STEP_TIGHTENING_NB * TIGHTENING_TRAIL_ENTRY_NB
    # Both starting sizes are a flat floor for the models the flat floor already covers, and a
    # model-derived one for the wide models it does not reach. The flat halves are measured: across the
    # 27 benchmark models the trail never exceeds 4096 entries and the stack never exceeds 64 rows, so
    # 65536 and 8192 leave every one of them growing exactly zero times -- raising them further would buy
    # nothing and give back the memory trailing was introduced to win.
    # The model-derived halves are where growth actually happens. A live trail runs between 2 and 12
    # times headroom on those same models, so 16 covers the observed band with margin; a stack deep
    # enough to matter is bounded by the decisions on the path, which scales with domain_nb. Both bind
    # only past a few thousand variables -- exactly where a doubling copies the most, and where the
    # allocation is small next to the triggers and propagator arrays a model that wide already carries.
    trail_log = np.empty((max(1 << 16, 16 * trail_headroom), 2), dtype=np.int32)
    trail_top = np.zeros((1,), dtype=np.int32)
    trail_indices = np.full(len(state), -1, dtype=np.int32)
    choice_point_stk = np.zeros((max(1 << 13, 4 * domain_nb), CHOICE_POINT_WIDTH), dtype=np.int32)
    choice_point_top = np.ones((1,), dtype=np.uint32)
    return (
        state,
        domains,
        entailed,
        trail_headroom,
        trail_log,
        trail_top,
        trail_indices,
        choice_point_stk,
        choice_point_top,
    )


def double_rows(array: NDArray) -> NDArray:
    """
    Returns a copy of an array with twice its rows, for the trail or the stack of choice points when it is full.

    The rows that follow the copy are not initialized: the search writes each row of these arrays before it reads
    it, and the rows it has written are all in the copy.

    :param array: the array, which keeps its type and the shape of its rows
    :type array: NDArray

    :return: the larger array
    :rtype: NDArray
    """
    grown = np.empty((2 * len(array), *array.shape[1:]), dtype=array.dtype)
    grown[: len(array)] = array
    return grown


def allocate_domain_buffer(offsets: NDArray) -> NDArray:
    """
    Allocates a reusable scratch buffer for prop_domains to avoid one allocation per propagator call.

    Sized to the largest propagator arity (which can exceed domain_nb when a propagator
    references the same variable twice, e.g. count_eq).
    Allocated once at solver init and threaded through the consistency algorithms.

    :param offsets: the CSR offsets delimiting each propagator's slice of propagator_variables
    :type offsets: NDArray

    :return: a scratch buffer sized to the maximal propagator arity
    :rtype: NDArray
    """
    max_arity = int(np.diff(offsets[:, OFFSETS_VARIABLE]).max(initial=0))
    return np.empty((max_arity, 2), dtype=np.int32)
