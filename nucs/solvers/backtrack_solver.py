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
from collections.abc import Callable, Iterable, Iterator

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
    DOM_HEURISTIC_MIN_VALUE,
    SIGN_DOM_HEURISTIC,
    SIGN_VAR_HEURISTIC,
    VAR_HEURISTIC_FCTS,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
)
from nucs.numba_helper import (
    NUMBA_DISABLE_JIT,
    ComputeDomainsFunctions,
    ConsistencyAlgorithmFunctions,
    DomainHeuristicFunctions,
    VariableHeuristicFunctions,
    build_function_ptrs,
)
from nucs.numpy_helper import flatten_arrays
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
    tighten_objective_at_root,
)
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_FCTS, SIGN_CONSISTENCY_ALG
from nucs.solvers.search import Search
from nucs.solvers.solver import OPTIM_RESET, Solver, get_solution
from nucs.statistics import (
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_ELAPSED_TIME,
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

# Trail entries a step of the search needs beyond one per cell of the backtrackable state.
# The barrier in trail_set trails each cell at most once per choice point, so a fixpoint cannot need more
# than len(state) entries however long it runs. The tightenings the search applies around it are not
# covered by that budget: each writes at a mark the trail holds nothing for yet, so every one of their
# writes is trailed. A tightening writes a domain's two bounds and, when it grounds the variable, the
# unbound count; a step applies at most two of them -- branch's decision is one, while backtracking a
# choice point applies its parked alternative and then the branch-and-bound objective bound.
TIGHTENING_TRAIL_ENTRY_NB = 3  # the two bounds of a domain and the unbound count
STEP_TIGHTENING_NB = 2  # an alternative then the objective bound, the longer of the two ways out of a step


class BacktrackSolver(Solver):
    """
    A solver relying on a backtracking mechanism.
    """

    # the per-search ragged collections threaded into solve_one_step, stored CSR-style: a flat concatenation
    # plus the offsets delimiting each search's slice (and, for the 2d parameter arrays, their shapes)
    decision_variables: NDArray
    decision_variables_offsets: NDArray
    var_heuristic_params: NDArray
    var_heuristic_params_offsets: NDArray
    var_heuristic_params_shapes: NDArray
    dom_heuristic_params: NDArray
    dom_heuristic_params_offsets: NDArray
    dom_heuristic_params_shapes: NDArray
    # the function tables threaded into solve_one_step (Numba typed lists under the JIT, plain Python lists otherwise)
    consistency_alg_fcts: ConsistencyAlgorithmFunctions
    var_heuristic_fcts: VariableHeuristicFunctions
    dom_heuristic_fcts: DomainHeuristicFunctions
    compute_domains_fcts: ComputeDomainsFunctions

    def __init__(
        self,
        problem: Problem,
        consistency_algorithm: int = CONSISTENCY_ALG_BC,
        decision_variables: Iterable[int] | None = None,
        var_heuristic: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
        var_heuristic_params: list[list[int]] | None = None,
        dom_heuristic: int = DOM_HEURISTIC_MIN_VALUE,
        dom_heuristic_params: list[list[int]] | None = None,
        searches: list[Search] | None = None,
        choice_point_max_height: int | None = None,
        trail_max_size: int | None = None,
        log_level: str = LOG_LEVEL_INFO,
    ):
        """
        Initializes the solver.

        :param problem: the problem to be solved
        :type problem: Problem
        :param consistency_algorithm: the consistency algorithm, defaults to bound consistency
        :type consistency_algorithm: int
        :param decision_variables: the variables on which decisions will be made, defaults to None
        :type decision_variables: Optional[Iterable[int]]
        :param var_heuristic: the heuristic for selecting a variable,
                              defaults to the first non instantiated
        :type var_heuristic: int
        :param var_heuristic_params: a list of lists of parameters,
                                     usually parameters are costs and there is a list of value costs per variable
        :type var_heuristic_params: Optional[List[List[int]]]
        :param dom_heuristic: the heuristic for reducing a domain,
                              defaults to instantiating the domain to its first value
        :type dom_heuristic: int
        :param dom_heuristic_params: a list of lists of parameters,
                                     usually parameters are costs and there is a list of value costs per variable
        :type dom_heuristic_params: Optional[List[List[int]]]
        :param searches: an ordered list of searches defining a sequential search; when None a single search
                         is built from the decision_variables / var_heuristic / dom_heuristic arguments above.
                         The union of the searches' decision variables should cover every branchable variable.
        :type searches: Optional[List[Search]]
        :param choice_point_max_height: the initial maximal height of the choice point stack, grown as needed,
                                        defaults to whichever is larger of 8192 and four rows per variable
        :type choice_point_max_height: Optional[int]
        :param trail_max_size: the initial maximal number of trail entries, grown as needed,
                               defaults to whichever is larger of 65536 and sixteen steps' worth of headroom
        :type trail_max_size: Optional[int]
        :param log_level: the log level,
                          defaults to INFO
        :type log_level: str
        """
        super().__init__(problem, log_level)
        if var_heuristic_params is None:
            var_heuristic_params = [[]]
        if dom_heuristic_params is None:
            dom_heuristic_params = [[]]
        if searches is None:
            searches = [
                Search(decision_variables, var_heuristic, var_heuristic_params, dom_heuristic, dom_heuristic_params)
            ]
        # every search keeps its own array of decision variables, variable/domain heuristic and parameters
        decision_variables_per_search: list[NDArray] = []
        var_heuristics: list[int] = []
        dom_heuristics: list[int] = []
        var_params: list[NDArray] = []
        dom_params: list[NDArray] = []
        for search in searches:
            search_vars = (
                list(range(problem.domain_nb)) if search.decision_variables is None else list(search.decision_variables)
            )
            decision_variables_per_search.append(np.array(search_vars, dtype=np.uint32))
            var_heuristics.append(search.var_heuristic)
            dom_heuristics.append(search.dom_heuristic)
            var_params.append(np.array(search.var_heuristic_params, dtype=np.int64))
            dom_params.append(np.array(search.dom_heuristic_params, dtype=np.int64))
        logger.info(f"BacktrackSolver uses decision domains {[dv.tolist() for dv in decision_variables_per_search]}")
        self.decision_variables, self.decision_variables_offsets = flatten_arrays(decision_variables_per_search)
        logger.info(f"BacktrackSolver uses variable heuristics {var_heuristics}")
        self.var_heuristic_params, self.var_heuristic_params_offsets = flatten_arrays(var_params)
        self.var_heuristic_params_shapes = np.array([params.shape for params in var_params], dtype=np.int64)
        logger.info(f"BacktrackSolver uses domain heuristics {dom_heuristics}")
        self.dom_heuristic_params, self.dom_heuristic_params_offsets = flatten_arrays(dom_params)
        self.dom_heuristic_params_shapes = np.array([params.shape for params in dom_params], dtype=np.int64)
        logger.info(f"BacktrackSolver uses consistency algorithm {consistency_algorithm}")
        self.triggered_propagators = buckets_create(problem.propagator_nb)
        self.domain_buffer = get_domain_buffer(problem.offsets)
        logger.debug("Initializing choice points")
        # all the backtrackable state in one flat int32 array, so that one undo log and one undo loop
        # restore every kind of it:
        #     [ 2 * domain_nb domain bounds | propagator_nb entailment flags | the unbound count ]
        # domains is a (domain_nb, 2) view of its head and entailed a view of the middle -- the same
        # memory, addressed the way each reader wants it -- so the flat index of (variable, bound) is
        # (variable << 1) | bound, and that of propagator p is 2 * domain_nb + p. Restoring a domain
        # bound and reactivating an entailed propagator are then the same instruction.
        domain_nb = self.problem.domain_nb
        propagator_nb = self.problem.propagator_nb
        propagator_entailment_offset = 2 * domain_nb
        unbound_count_offset = propagator_entailment_offset + propagator_nb
        self.state = np.zeros(unbound_count_offset + 1, dtype=np.int32)
        self.domains = self.state[:propagator_entailment_offset].reshape(domain_nb, 2)
        self.entailed = self.state[propagator_entailment_offset:unbound_count_offset]
        # the guard lets a choice point trail each cell of state at most once -- every domain bound, every
        # entailment flag and the count -- so a fixpoint cannot need more than len(state) entries, whatever
        # it does; the tightenings the search applies around it write at their own mark and are counted on
        # top. The solver grows the trail when this much room is no longer there.
        self.trail_headroom = len(self.state) + STEP_TIGHTENING_NB * TIGHTENING_TRAIL_ENTRY_NB
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
        self.trail_log = np.empty((trail_max_size or max(1 << 16, 16 * self.trail_headroom), 2), dtype=np.int32)
        self.trail_top = np.zeros((1,), dtype=np.int32)
        self.trail_indices = np.full(len(self.state), -1, dtype=np.int32)
        self.choice_point_stk = np.zeros(
            (choice_point_max_height or max(1 << 13, 4 * domain_nb), CHOICE_POINT_WIDTH), dtype=np.int32
        )
        self.choice_point_top = np.ones((1,), dtype=np.uint32)
        # the branch-and-bound bound, as OBJECTIVE_VARIABLE, OBJECTIVE_BOUND and OBJECTIVE_VALUE: the variable
        # optimized, the side of its domain to tighten, and the best value found so far. It is solver
        # state, not choice-point state -- the bound holds for the whole remaining search, so backtrack
        # re-applies it to each choice point it resumes rather than it being written into them all when it
        # is found, and nothing about it is trailed. OBJECTIVE_VARIABLE stays -1 outside OPTIM_PRUNE, which is
        # how backtrack knows there is no bound to apply: OPTIM_RESET tightens at the root instead.
        self.objective = np.full(OBJECTIVE_WIDTH, -1, dtype=np.int32)
        logger.info(
            f"The stack of choice points starts at {len(self.choice_point_stk)} rows and grows when it runs out"
        )
        logger.info(f"The trail starts at {len(self.trail_log)} entries and grows when it runs out")
        self._choice_point_init()
        logger.debug("Choice points initialized")
        logger.debug("Initializing statistics")
        self.statistics = statistics_init(get_algorithm_nb())
        logger.debug("Statistics initialized")
        if NUMBA_DISABLE_JIT:
            self.compute_domains_fcts = COMPUTE_DOMAINS_FCTS
            self.consistency_alg_fcts = [CONSISTENCY_ALG_FCTS[consistency_algorithm]]
            self.var_heuristic_fcts = [VAR_HEURISTIC_FCTS[h] for h in var_heuristics]
            self.dom_heuristic_fcts = [DOM_HEURISTIC_FCTS[h] for h in dom_heuristics]
        else:
            # resolving only the algorithms used by the problem keeps the init cost proportional
            # to the problem instead of the whole propagator library
            self.compute_domains_fcts = build_function_ptrs(
                COMPUTE_DOMAINS_FCTS, SIGN_COMPUTE_DOMAINS, np.unique(self.problem.algorithms), ALG_DUMMY
            )
            self.consistency_alg_fcts = build_function_ptrs(
                [CONSISTENCY_ALG_FCTS[consistency_algorithm]], SIGN_CONSISTENCY_ALG
            )
            self.var_heuristic_fcts = build_function_ptrs(
                [VAR_HEURISTIC_FCTS[h] for h in var_heuristics], SIGN_VAR_HEURISTIC
            )
            self.dom_heuristic_fcts = build_function_ptrs(
                [DOM_HEURISTIC_FCTS[h] for h in dom_heuristics], SIGN_DOM_HEURISTIC
            )
        logger.debug("BacktrackSolver initialized")

    def solve(self, timeout: float | None = None) -> Iterator[NDArray]:
        logger.info("Solving and iterating over the solutions")
        for solution in self._iterate_solutions(lambda _: self._backtrack(), timeout):
            logger.debug("Found a solution")
            yield solution

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
        deadline = None if timeout is None else time.monotonic() + timeout
        t0 = time.perf_counter_ns()
        buckets_empty(self.triggered_propagators, self.problem.priorities)
        buckets_init(self.triggered_propagators, self.problem.priorities)
        # no incumbent yet, so the first descent runs unbounded; _advance_after_optimum arms the objective.
        # An enumeration disarms it here too: the solver may have been optimized with before.
        self.objective[OBJECTIVE_VARIABLE] = -1
        while (solution := self._solve_one()) is not None:
            self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0
            yield solution
            t0 = time.perf_counter_ns()
            if self._expired(deadline):
                break
            if not advance(solution):
                break
        self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0

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
            self._grow(status)

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
            self.decision_variables,
            self.decision_variables_offsets,
            self.var_heuristic_fcts,
            self.var_heuristic_params,
            self.var_heuristic_params_offsets,
            self.var_heuristic_params_shapes,
            self.dom_heuristic_fcts,
            self.dom_heuristic_params,
            self.dom_heuristic_params_offsets,
            self.dom_heuristic_params_shapes,
            self.compute_domains_fcts,
            self.domain_buffer,
            self.problem.idempotencies,
            self.objective,
            self.trail_headroom,
        )

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
        if mode == OPTIM_RESET:
            logger.debug("Resetting solver")
            self._choice_point_init()
            if not tighten_objective_at_root(
                self.state, self.trail_log, self.trail_top, self.trail_indices, variable, value, bound
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

    def get_statistics_as_array(self) -> NDArray:
        """
        Returns the statistics as a Numpy array.

        :return: the statistics array
        :rtype: NDArray
        """
        return self.statistics

    def get_statistics_as_dictionary(self) -> dict[str, int]:
        """
        Returns the statistics as a dictionary.

        :return: a dictionary mapping statistic labels to values
        :rtype: Dict[str, int]
        """
        return statistics_as_dictionary(self.statistics, get_algorithm_names())


@njit(cache=True)
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
    compute_domains_fcts: ComputeDomainsFunctions,
    domain_buffer: NDArray,
    idempotencies: NDArray,
    objective: NDArray,
    trail_headroom: int,
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
    :param compute_domains_fcts: the typed list of compute_domains functions, built once at solver init
    :type compute_domains_fcts: ComputeDomainsFcts
    :param domain_buffer: a scratch buffer for prop_domains,
                          sized to max propagator arity, allocated once at solver init
    :type domain_buffer: NDArray
    :param idempotencies: whether each algorithm reaches its own fixpoint in a single call, indexed by
                          algorithm rather than by propagator
    :type idempotencies: NDArray
    :param objective: the objective as a Numpy array of variable, bound and value,
                      whose variable is -1 when not optimizing
    :type objective: NDArray
    :param trail_headroom: the trail entries any one step of the search can need
    :type trail_headroom: int

    :return: why the step returned, and the solution when it found one
    :rtype: Tuple[int, Optional[NDArray]]
    """
    consistency_alg_fct = consistency_alg_fcts[0]
    nb_searches = len(decision_variables_offsets) - 1
    max_choice_point = len(choice_point_stk) - 3  # a ternary split pushes two choice points and marks a third
    while True:
        # the arrays are caller-allocated, so the search stops for the solver to grow one rather than
        # overrun it silently -- with boundscheck off, the overrun is what would otherwise happen
        if trail_top[0] + trail_headroom > len(trail):
            return SOLVER_TRAIL_FULL, None
        if choice_point_top[0] > max_choice_point:
            return SOLVER_CHOICE_POINTS_FULL, None
        problem_status = consistency_alg_fct(
            statistics,
            idempotencies,
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
            compute_domains_fcts,
            domain_buffer,
        )
        choice_point = choice_point_top[0]
        if problem_status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLUTION_NB] += 1
            return SOLVER_RUNNING, get_solution(domains)
        branched = False
        if problem_status == PROBLEM_UNBOUND:
            # sequential search: the first search that still has an unbound decision variable owns the
            # decision and branches with its own variable and domain heuristics
            for search_idx in range(nb_searches):
                variable = var_heuristic_fcts[search_idx](
                    decision_variables[
                        decision_variables_offsets[search_idx] : decision_variables_offsets[search_idx + 1]
                    ],
                    domains,
                    var_heuristic_params[
                        var_heuristic_params_offsets[search_idx] : var_heuristic_params_offsets[search_idx + 1]
                    ].reshape(var_heuristic_params_shapes[search_idx, 0], var_heuristic_params_shapes[search_idx, 1]),
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
                        triggered_propagators,
                        entailed,
                        triggers,
                        triggers_offsets,
                        priorities,
                        variable,
                        events,
                    )
                    statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
                    statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = max(
                        statistics[STATS_IDX_SOLVER_CHOICE_DEPTH], choice_point
                    )
                    branched = True
                    break
        # either the problem is inconsistent, or no search can claim a variable although variables remain
        # unbound -- a choice point whose domains admit no assignment. Both are dead ends: backtrack.
        if not branched and not backtrack(
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
