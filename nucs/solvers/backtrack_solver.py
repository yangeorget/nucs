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
from collections.abc import Callable, Iterator

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import buckets_create, buckets_empty, buckets_init
from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    LOG_LEVEL_INFO,
    OBJECTIVE_BOUND,
    OBJECTIVE_VALUE,
    OBJECTIVE_VARIABLE,
    OBJECTIVE_WIDTH,
)
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_FCTS,
    SIGN_DOM_HEURISTIC,
    SIGN_VAR_HEURISTIC,
    VAR_HEURISTIC_FCTS,
)
from nucs.numba_helper import (
    NUMBA_DISABLE_JIT,
    ConsistencyAlgorithmFunctions,
    DomainHeuristicFunctions,
    VariableHeuristicFunctions,
    addresses_from_functions,
    build_function_ptrs,
)
from nucs.problems.problem import OFFSETS_VARIABLE, PROBLEM_BOUND, PROBLEM_UNBOUND, Problem
from nucs.propagators.propagators import (
    ALG_DUMMY,
    COMPUTE_DOMAINS_FCTS,
    SIGN_COMPUTE_DOMAINS,
    get_algorithm_names,
    get_algorithm_nb,
    update_propagators,
)
from nucs.solvers.choice_points import (
    CHOICE_POINT_WIDTH,
    backtrack,
    branch,
    choice_point_init,
    tighten_objective,
)
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_FCTS, SIGN_CONSISTENCY_ALG
from nucs.solvers.interruption import INTERRUPTION_DEADLINE, Interruption
from nucs.solvers.restarts import RESTART_NONE, restart_limits
from nucs.solvers.search import Search, flatten_searches
from nucs.solvers.search_arrays import allocate_search_arrays
from nucs.solvers.solver import OPTIM_RESET, Solver, get_solution
from nucs.solvers.weights import weights_init
from nucs.statistics import (
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_ELAPSED_TIME,
    STATS_IDX_SOLVER_RESTART_NB,
    statistics_as_dictionary,
    statistics_init,
)

logger = logging.getLogger(__name__)

# Capacity outcomes of a search step.
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


class BacktrackSolver(Solver):
    """
    A solver relying on a backtracking mechanism.
    """

    # the function tables threaded into solve_one_step (Numba typed lists under the JIT, plain Python lists otherwise)
    consistency_alg_fcts: ConsistencyAlgorithmFunctions
    var_heuristic_fcts: VariableHeuristicFunctions
    dom_heuristic_fcts: DomainHeuristicFunctions
    # the compiled compute_domains address of each algorithm, which the consistency algorithm calls a propagator through
    compute_domains_addrs: NDArray

    def __init__(
        self,
        problem: Problem,
        consistency_algorithm: int = CONSISTENCY_ALG_BC,
        searches: list[Search] | None = None,
        log_level: str = LOG_LEVEL_INFO,
        weight_decay: float = 1.0,
        restart_policy: str = RESTART_NONE,
        restart_scale: int = 100,
        restart_base: float = 1.5,
        last_conflict: bool = False,
    ):
        """
        Initializes the solver.

        :param problem: the problem to be solved
        :type problem: Problem
        :param consistency_algorithm: the consistency algorithm, defaults to bound consistency
        :type consistency_algorithm: int
        :param searches: an ordered list of searches defining a sequential search, each with its decision variables,
                         its variable and domain heuristics and their parameters; defaults to one search that
                         branches on every variable with the default heuristics of Search. The union of the
                         searches' decision variables should cover every branchable variable.
        :type searches: Optional[List[Search]]
        :param log_level: the log level,
                          defaults to INFO
        :type log_level: str
        :param weight_decay: in (0, 1], how much the failure weight of a propagator keeps of a failure after each
                             later failure; 1 counts all the failures the same (dom/wdeg), less than 1 prefers the
                             recent ones (AFC), defaults to 1
        :type weight_decay: float
        :param restart_policy: one of RESTART_POLICIES (nucs.solvers.restarts): after how many failures each descent
                               restarts from the root, defaults to RESTART_NONE. When enumerating, the restarts stop
                               at the first solution, so that no solution is found twice.
        :type restart_policy: str
        :param restart_scale: the number of failures the restart policy multiplies, defaults to 100
        :type restart_scale: int
        :param restart_base: the ratio of the geometric restart policy, defaults to 1.5
        :type restart_base: float
        :param last_conflict: whether the search branches first on the variable of the last refuted decision, as long
                              as it is unbound (last-conflict reasoning), defaults to False
        :type last_conflict: bool
        """
        super().__init__(problem, log_level)
        if searches is None:
            searches = [Search()]
        # every search keeps its own decision variables, variable and domain heuristics and their parameters
        self.flat_searches = flatten_searches(searches, problem.domain_nb)
        logger.info(f"BacktrackSolver uses {self.flat_searches}")
        logger.info(f"BacktrackSolver uses consistency algorithm {consistency_algorithm}")
        self.triggered_propagators = buckets_create(problem.propagator_nb)
        self.domain_buffer = get_domain_buffer(problem.offsets)
        logger.debug("Initializing choice points")
        arrays = allocate_search_arrays(problem)
        self.state, self.domains, self.entailed = arrays.state, arrays.domains, arrays.entailed
        self.trail_headroom, self.trail_log = arrays.trail_headroom, arrays.trail_log
        self.trail_top, self.trail_indices = arrays.trail_top, arrays.trail_indices
        self.choice_point_stk, self.choice_point_top = arrays.choice_point_stk, arrays.choice_point_top
        # the branch-and-bound bound, as OBJECTIVE_VARIABLE, OBJECTIVE_BOUND and OBJECTIVE_VALUE: the variable
        # optimized, the side of its domain to tighten, and the best value found so far. It is solver
        # state, not choice-point state -- the bound holds for the whole remaining search, so backtrack
        # re-applies it to each choice point it resumes rather than it being written into them all when it
        # is found, and nothing about it is trailed. OBJECTIVE_VARIABLE stays -1 outside OPTIM_PRUNE, which is
        # how backtrack knows there is no bound to apply: OPTIM_RESET tightens at the root instead.
        self.objective = np.full(OBJECTIVE_WIDTH, -1, dtype=np.int32)
        # read by solve_one_step at every node, and written by interrupt() and by the deadline of a search
        self.interruption = Interruption()
        logger.info(
            f"The stack of choice points starts at {len(self.choice_point_stk)} rows and grows when it runs out"
        )
        logger.info(f"The trail starts at {len(self.trail_log)} entries and grows when it runs out")
        self._choice_point_init()
        logger.debug("Choice points initialized")
        logger.debug("Initializing statistics")
        self.statistics = statistics_init(get_algorithm_nb())
        # the failure weight of each propagator: global, never trailed and never reset, so that what one solve
        # learns about the constraints serves the next one too
        self.propagator_weights = weights_init(problem.propagator_nb, weight_decay)
        # checked here, so that a wrong policy fails at construction rather than at the first solve
        restart_limits(restart_policy, restart_scale, restart_base)
        self.restart_policy = restart_policy
        self.restart_scale = restart_scale
        self.restart_base = restart_base
        self.restart_limits: Iterator[int] = iter(())
        self.search_control = np.full(SEARCH_CONTROL_WIDTH, -1, dtype=np.int64)
        self.search_control[SEARCH_CONTROL_LAST_CONFLICT] = int(last_conflict)
        # the objective of the best solution so far, (variable, value, bound) as in self.objective, which a restart
        # re-applies at the root
        self.restart_objective: tuple[int, int, int] | None = None
        logger.debug("Statistics initialized")
        # resolving only the algorithms used by the problem keeps the init cost proportional to the problem instead
        # of the whole propagator library; without the JIT this is a placeholder that call_compute_domains ignores
        self.compute_domains_addrs = addresses_from_functions(
            COMPUTE_DOMAINS_FCTS, SIGN_COMPUTE_DOMAINS, np.unique(self.problem.algorithms), ALG_DUMMY
        )
        if NUMBA_DISABLE_JIT:
            self.consistency_alg_fcts = [CONSISTENCY_ALG_FCTS[consistency_algorithm]]
            self.var_heuristic_fcts = [VAR_HEURISTIC_FCTS[h] for h in self.flat_searches.var_heuristics]
            self.dom_heuristic_fcts = [DOM_HEURISTIC_FCTS[h] for h in self.flat_searches.dom_heuristics]
        else:
            self.consistency_alg_fcts = build_function_ptrs(
                [CONSISTENCY_ALG_FCTS[consistency_algorithm]], SIGN_CONSISTENCY_ALG
            )
            self.var_heuristic_fcts = build_function_ptrs(
                [VAR_HEURISTIC_FCTS[h] for h in self.flat_searches.var_heuristics], SIGN_VAR_HEURISTIC
            )
            self.dom_heuristic_fcts = build_function_ptrs(
                [DOM_HEURISTIC_FCTS[h] for h in self.flat_searches.dom_heuristics], SIGN_DOM_HEURISTIC
            )
        logger.debug("BacktrackSolver initialized")

    def solve(self, timeout: float | None = None) -> Iterator[NDArray]:
        logger.info("Solving and iterating over the solutions")
        for solution in self._iterate_solutions(lambda _: self._backtrack_after_solution(), timeout):
            logger.debug("Found a solution")
            yield solution

    def _backtrack_after_solution(self) -> bool:
        """
        Moves the enumeration on from a solution, and stops the restarts: a restart would find the solutions
        emitted so far again.

        :return: whether the search can continue
        :rtype: bool
        """
        self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = -1
        return self._backtrack()

    def optimize(self, variable: int, bound: int, mode: str, timeout: float | None = None) -> Iterator[NDArray]:
        logger.info("Optimizing and iterating over the solutions")
        # minimizing a variable means tightening the DOMAIN_MAX side of its domain, and vice versa
        objective_bound = DOMAIN_MAX if bound == DOMAIN_MIN else DOMAIN_MIN
        for solution in self._iterate_solutions(
            lambda found: self._advance_after_optimum(variable, found[variable], objective_bound, mode),
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
        self.timed_out = False
        disarm = self.interruption.arm_deadline(timeout)
        try:
            t0 = time.perf_counter_ns()
            buckets_empty(self.triggered_propagators, self.problem.priorities)
            buckets_init(self.triggered_propagators, self.problem.priorities)
            # no solution yet, so the search has no objective bound; _advance_after_optimum arms it.
            # An enumeration disarms it here too: the solver may have been optimized with before.
            self.objective[OBJECTIVE_VARIABLE] = -1
            self.restart_objective = None
            self.restart_limits = restart_limits(self.restart_policy, self.restart_scale, self.restart_base)
            self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = next(self.restart_limits)
            self.search_control[SEARCH_CONTROL_FAILURE_NB] = 0
            self.search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = -1
            self.search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
            while (solution := self._solve_one()) is not None:
                self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0
                yield solution
                t0 = time.perf_counter_ns()
                # no check of the budget here: the timer stops the next call of solve_one_step at its first node,
                # and a search that advance() finds over has nothing left that a timeout could have cut
                if not advance(solution):
                    break
            self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0
        finally:
            # also when the consumer abandons the iteration, so that the timer cannot stop a later search
            disarm()

    def _solve_one(self) -> NDArray | None:
        """
        Searches for the next solution, growing whichever caller-allocated array runs out and resuming.

        :return: the next solution if it exists or None
        :rtype: Optional[NDArray]
        """
        while True:
            status, solution = self._solve_one_step()
            if status == SOLVER_RUNNING:
                return solution
            if status == SOLVER_INTERRUPTED:
                if self.interruption.cell[0] == INTERRUPTION_DEADLINE:
                    logger.info("Timeout reached, stopping the search")
                else:
                    logger.info("Interrupted, stopping the search")
                self.timed_out = True
                return None
            if status == SOLVER_RESTART:
                if not self._restart():
                    return None
                continue
            self._grow(status)

    def _restart(self) -> bool:
        """
        Restarts the search from the root, with the next failure limit of the restart policy.

        What the search learned stays: the propagator weights are not reset. The best solution so far is re-applied
        at the root, with a mark of 0 as OPTIM_RESET does, since the reset undoes every tightening above it.

        :return: false when the root cannot hold a better solution, which proves the best solution so far optimal
        :rtype: bool
        """
        logger.debug("Restarting")
        self.statistics[STATS_IDX_SOLVER_RESTART_NB] += 1
        self._choice_point_init()
        self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = next(self.restart_limits)
        self.search_control[SEARCH_CONTROL_FAILURE_NB] = 0
        self.search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = -1
        self.search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
        if self.restart_objective is not None:
            variable, value, bound = self.restart_objective
            if (
                tighten_objective(
                    self.state, self.trail_log, self.trail_top, self.trail_indices, 0, variable, value, bound
                )
                < 0
            ):
                return False
        # the descent ended at a failure: the failed filtering left propagators in the queue, and backtrack added
        # more
        buckets_empty(self.triggered_propagators, self.problem.priorities)
        buckets_init(self.triggered_propagators, self.problem.priorities)
        return True

    def _solve_one_step(self) -> tuple[int, NDArray | None]:
        """
        Runs one step of the search by forwarding the solver state to the jitted solve_one_step.

        :return: why the step returned, and the solution when it found one
        :rtype: Tuple[int, Optional[NDArray]]
        """
        return solve_one_step(
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
            self.objective,
            self.trail_headroom,
            self.interruption.cell,
            self.propagator_weights,
            self.search_control,
            self.flat_searches.variable_searches,
        )

    def interrupt(self) -> None:
        """
        Stops the search at its next node, as if its budget had run out: the iteration ends and
        :attr:`timed_out` is set. The interruption is final, every later search stops at once too.

        Safe to call from another thread or from a signal handler: it takes no lock, it only sets a flag and
        writes a cell that the compiled search reads, and solve_one_step releases the GIL so that such a thread
        gets to run.
        """
        self.interruption.interrupt()

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
        self.restart_objective = (variable, value, bound)
        if mode == OPTIM_RESET:
            logger.debug("Resetting solver")
            self._choice_point_init()
            # a root reset, as a restart is: the failures count from here and the conflict variable is forgotten
            self.search_control[SEARCH_CONTROL_FAILURE_NB] = 0
            self.search_control[SEARCH_CONTROL_CONFLICT_VARIABLE] = -1
            self.search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
            # at the root, so with a mark of 0: what this tightening writes is undone only by the next reset
            if (
                tighten_objective(
                    self.state, self.trail_log, self.trail_top, self.trail_indices, 0, variable, value, bound
                )
                < 0
            ):
                return False
            buckets_init(self.triggered_propagators, self.problem.priorities)
        else:
            logger.debug("Pruning choice points")
            # arm the objective and let backtrack apply it: the bound is re-applied to each choice point the
            # search resumes, so the choice points it kills are dropped as they are reached rather than up front
            self.objective[OBJECTIVE_VARIABLE] = variable
            self.objective[OBJECTIVE_BOUND] = bound
            self.objective[OBJECTIVE_VALUE] = value
            if not self._backtrack():
                return False
        return True

    def _choice_point_init(self) -> None:
        """
        Resets the search to the root.
        """
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

    def _grow(self, status: int) -> None:
        """
        Doubles whichever caller-allocated array the search ran out of, and lets it continue.

        :param status: SOLVER_TRAIL_FULL or SOLVER_CHOICE_POINTS_FULL, the array that filled up
        :type status: int

        Nothing of the search is lost. The trail keeps its contents, so every mark and every position in
        trail_indices still addresses the same entry; the choice point stack keeps its rows. Sizing either array for its
        worst case instead -- depth x (2 x domain_nb + 1) trail entries -- would hand back the memory
        this representation wins, and a hard failure would end a long optimization run for no reason.
        """
        if status == SOLVER_TRAIL_FULL:
            trail = np.empty((2 * len(self.trail_log), 2), dtype=np.int32)
            trail[: len(self.trail_log)] = self.trail_log
            self.trail_log = trail
            logger.info(f"The trail grew to {len(self.trail_log)} entries")
        else:
            choice_point_stk = np.zeros((2 * len(self.choice_point_stk), CHOICE_POINT_WIDTH), dtype=np.int32)
            choice_point_stk[: len(self.choice_point_stk)] = self.choice_point_stk
            self.choice_point_stk = choice_point_stk
            logger.info(f"The stack of choice points grew to a maximal height of {len(self.choice_point_stk)}")

    def _backtrack(self) -> bool:
        """
        Backtracks by forwarding the solver state to the jitted backtrack.

        :return: true iff it was possible to backtrack
        :rtype: bool
        """
        # the search resumes on a refutation, and a failure that follows one does not name a conflict variable
        self.search_control[SEARCH_CONTROL_DECISION_VARIABLE] = -1
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
            self.objective,
        )

    def get_statistics_as_dictionary(self) -> dict[str, int]:
        """
        Returns the statistics as a dictionary.

        :return: a dictionary mapping statistic labels to values
        :rtype: Dict[str, int]
        """
        return statistics_as_dictionary(self.statistics, get_algorithm_names())


@njit(cache=True, nogil=True)
def solve_one_step(
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
    objective: NDArray,
    trail_headroom: int,
    interruption: NDArray,
    propagator_weights: NDArray,
    search_control: NDArray,
    variable_searches: NDArray,
) -> tuple[int, NDArray | None]:
    """
    Searches for one solution, stopping early when an array it cannot grow runs out of room.

    It is a step rather than the whole search because the trail and the choice point stack belong to the
    caller: when either fills up, the returned status names it and no solution comes back, for _solve_one
    to grow that array and call again. Nothing of the search is lost in between.

    The status is returned rather than written into a one-cell array because nothing inside @njit reads
    it -- it is the reason the step stopped, and only the Python caller acts on it. A solution alone
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
    :type consistency_alg_fcts: ConsistencyAlgFcts
    :param decision_variables: the concatenation of the per-search decision variable arrays
    :type decision_variables: NDArray
    :param decision_variables_offsets: the CSR offsets delimiting each search's slice of decision_variables
    :type decision_variables_offsets: NDArray
    :param var_heuristic_fcts: the typed list of variable heuristic functions, one per search
    :type var_heuristic_fcts: VarHeuristicFcts
    :param var_heuristic_params: the flattened concatenation of the per-search variable heuristic parameter arrays
    :type var_heuristic_params: NDArray
    :param var_heuristic_params_offsets: the CSR offsets delimiting each search's slice of var_heuristic_params
    :type var_heuristic_params_offsets: NDArray
    :param var_heuristic_params_shapes: the 2d shape of each search's variable heuristic parameter array
    :type var_heuristic_params_shapes: NDArray
    :param dom_heuristic_fcts: the typed list of domain heuristic functions, one per search
    :type dom_heuristic_fcts: DomHeuristicFcts
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
    :param objective: the objective as a Numpy array of variable, bound and value,
                      whose variable is -1 when not optimizing
    :type objective: NDArray
    :param trail_headroom: the trail entries any one step of the search can need
    :type trail_headroom: int
    :param interruption: the one-cell array of an Interruption, non-zero once the search is asked to stop
    :type interruption: NDArray
    :param propagator_weights: the failure weight of each propagator, followed by the increment and its growth
                               (see nucs.solvers.weights)
    :type propagator_weights: NDArray
    :param search_control: the state of the restarts and of last-conflict, see SEARCH_CONTROL_*
    :type search_control: NDArray
    :param variable_searches: the search that owns each variable, -1 for a variable no search branches on
    :type variable_searches: NDArray

    :return: why the step returned, and the solution when it found one
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
        restart_limit = search_control[SEARCH_CONTROL_RESTART_LIMIT]
        if 0 <= restart_limit <= search_control[SEARCH_CONTROL_FAILURE_NB]:
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
                        triggered_propagators, entailed, triggers, triggers_offsets, priorities, variable, events
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
                objective,
            ):
                return SOLVER_RUNNING, None


def get_domain_buffer(offsets: NDArray) -> NDArray:
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
    max_arity = np.int64(0)
    for propagator_idx in range(len(offsets) - 1):
        arity = np.int64(offsets[propagator_idx + 1, OFFSETS_VARIABLE] - offsets[propagator_idx, OFFSETS_VARIABLE])
        max_arity = max(max_arity, arity)
    return np.empty((max_arity, 2), dtype=np.int32)
