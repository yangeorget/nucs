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
from collections.abc import Iterable, Iterator

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from nucs.buckets import buckets_create, buckets_empty, buckets_init
from nucs.constants import (
    DECISION_VALUE,
    DECISION_WIDTH,
    LEVEL_WIDTH,
    LOG_LEVEL_INFO,
    MAX,
    MIN,
    NUMBA_DISABLE_JIT,
    OBJ_BOUND,
    OBJ_VALUE,
    OBJ_VARIABLE,
    OBJ_WIDTH,
    OPTIM_RESET,
    PROBLEM_BOUND,
    PROBLEM_UNBOUND,
    SIGN_COMPUTE_DOMAINS,
    SIGN_CONSISTENCY_ALG,
    SIGN_DOM_HEURISTIC,
    SIGN_VAR_HEURISTIC,
    SOLVER_LEVELS_FULL,
    SOLVER_RUNNING,
    SOLVER_STATUS,
    SOLVER_STATUS_WIDTH,
    SOLVER_TRAIL_FULL,
    STATS_ALG_IDX_FILTER_NB,
    STATS_ALG_IDX_FILTER_NO_CHANGE_NB,
    STATS_ALG_WIDTH,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_ELAPSED_TIME,
    STATS_LBL_ALG_BC_NB,
    STATS_LBL_PROPAGATOR_ENTAILMENT_NB,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_INCONSISTENCY_NB,
    STATS_LBL_SOLUTION_NB,
    STATS_LBL_SOLVER_BACKTRACK_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
    STATS_LBL_SOLVER_CHOICE_NB,
    STATS_LBL_SOLVER_ELAPSED_TIME,
    STATS_MAX,
    VARIABLE,
)
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_FCTS,
    DOM_HEURISTIC_MIN_VALUE,
    VAR_HEURISTIC_FCTS,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
)
from nucs.numba_helper import (
    ComputeDomainsFunctions,
    ConsistencyAlgorithmFunctions,
    DomainHeuristicFunctions,
    VariableHeuristicFunctions,
    build_function_ptrs,
)
from nucs.numpy_helper import flatten_arrays
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    ALG_DUMMY,
    COMPUTE_DOMAINS_FCTS,
    IDEMPOTENT,
    get_algorithm_nb,
    update_propagators,
)
from nucs.solvers.choice_points import backtrack, branch, cp_init, fix_choice_point
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_BC, CONSISTENCY_ALG_FCTS
from nucs.solvers.search import Search
from nucs.solvers.solver import Solver, get_solution

logger = logging.getLogger(__name__)


class BacktrackSolver(Solver):
    """
    A solver relying on a backtracking mechanism.
    """

    # the per-search ragged collections threaded into solve_one, stored CSR-style: a flat concatenation
    # plus the offsets delimiting each search's slice (and, for the 2d parameter arrays, their shapes)
    decision_variables: NDArray
    decision_variables_offsets: NDArray
    var_heuristic_params: NDArray
    var_heuristic_params_offsets: NDArray
    var_heuristic_params_shapes: NDArray
    dom_heuristic_params: NDArray
    dom_heuristic_params_offsets: NDArray
    dom_heuristic_params_shapes: NDArray
    # the function tables threaded into solve_one (Numba typed lists under the JIT, plain Python lists otherwise)
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
        stks_max_height: int = 8192,
        trail_max_size: int = 1 << 16,
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
        :param stks_max_height: the initial maximal height of the choice point stacks, grown as needed,
                                defaults to 8192
        :type stks_max_height: int
        :param trail_max_size: the initial maximal number of trail entries, grown as needed,
                               defaults to 1 << 16
        :type trail_max_size: int
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
        # restore every kind of it: [ 2 * domain_nb domain bounds | the unbound-variable count ].
        # domains is a (domain_nb, 2) view of its head -- the same memory, addressed the way every
        # reader wants it -- so the flat index of (variable, bound) is (variable << 1) | bound.
        domain_nb = self.problem.domain_nb
        self.state = np.zeros(2 * domain_nb + 1, dtype=np.int32)
        self.domains = self.state[: 2 * domain_nb].reshape(domain_nb, 2)
        self.trail = np.empty((trail_max_size, 2), dtype=np.int32)
        self.trail_top = np.zeros((1,), dtype=np.int32)
        self.pos = np.full(2 * domain_nb + 1, -1, dtype=np.int32)
        self.level_stk = np.zeros((stks_max_height, LEVEL_WIDTH), dtype=np.int32)
        self.stks_top = np.ones((1,), dtype=np.uint32)
        self.status = np.zeros(SOLVER_STATUS_WIDTH, dtype=np.int32)
        # a filtering can trail every cell of a level once and no more, so this much headroom is enough
        # for any single step of the search; the solver grows the trail when it runs out
        self.trail_headroom = 2 * domain_nb + 8
        # entailment is tracked by a trail rather than a per-level array: entailed_propagator_depths[p]
        # holds the depth at which propagator p was entailed (-1 when active), entailment_trail records the
        # entailed propagators in order (its first cell is the trail size) so backtracking can reactivate them
        self.entailed_propagator_depths = np.empty(self.problem.propagator_nb, dtype=np.int32)
        self.entailment_trail = np.empty(self.problem.propagator_nb + 1, dtype=np.int32)
        # the branch-and-bound bound is solver state, not choice-point state: OBJ_VARIABLE is -1 outside
        # OPTIM_PRUNE, and backtrack re-applies the bound to each level it resumes
        self.objective = np.full(OBJ_WIDTH, -1, dtype=np.int32)
        # scratch for the domain heuristic's split value, allocated once rather than returned as a tuple
        self.decision = np.zeros(DECISION_WIDTH, dtype=np.int32)
        logger.info(f"The stacks of the choice points have a maximal height of {stks_max_height}")
        logger.info(f"The trail has a maximal size of {trail_max_size} entries, and grows when it runs out")
        self.initial_domains = np.array(problem.domains)
        self._cp_init()
        logger.debug("Choice points initialized")
        logger.debug("Initializing statistics")
        self.statistics = np.zeros(STATS_MAX + STATS_ALG_WIDTH * get_algorithm_nb(), dtype=np.int64)
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
        self.timed_out = False
        deadline = None if timeout is None else time.monotonic() + timeout
        t0 = time.perf_counter_ns()
        buckets_empty(self.triggered_propagators, self.problem.priorities)
        buckets_init(self.triggered_propagators, self.problem.priorities)
        self.objective[OBJ_VARIABLE] = -1
        while True:
            solution = self._solve_one()
            if solution is None:
                break
            self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0
            logger.debug("Found a solution")
            yield solution
            t0 = time.perf_counter_ns()
            if self._expired(deadline):
                break
            if not self._backtrack():
                break
        self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0

    def optimize(self, variable: int, bound: int, mode: str, timeout: float | None = None) -> Iterator[NDArray]:
        logger.info("Optimizing and iterating over the solutions")
        self.timed_out = False
        deadline = None if timeout is None else time.monotonic() + timeout
        t0 = time.perf_counter_ns()
        buckets_empty(self.triggered_propagators, self.problem.priorities)
        buckets_init(self.triggered_propagators, self.problem.priorities)
        # no incumbent yet, so the first descent runs unbounded; _advance_after_optimum arms the objective
        self.objective[OBJ_VARIABLE] = -1
        while (solution := self._solve_one()) is not None:
            self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0
            logger.info(f"Found a local optimum: {solution[variable]}")
            yield solution
            t0 = time.perf_counter_ns()
            if self._expired(deadline):
                break
            # minimizing a variable means tightening the MAX side of its domain, and vice versa
            if not self._advance_after_optimum(variable, solution[variable], MAX if bound == MIN else MIN, mode):
                break
        self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME] += time.perf_counter_ns() - t0

    def _solve_one(self) -> NDArray | None:
        """
        Searches for the next solution by forwarding the solver state to the jitted solve_one.

        :return: the next solution if it exists or None
        :rtype: Optional[NDArray]
        """
        while True:
            solution = self._solve_one_step()
            if self.status[SOLVER_STATUS] == SOLVER_RUNNING:
                return solution
            self._grow()

    def _solve_one_step(self) -> NDArray | None:
        """
        Searches for the next solution until it is found, the search is exhausted, or a stack fills up.

        :return: the next solution if it exists or None
        :rtype: Optional[NDArray]
        """
        return solve_one(
            self.problem.propagator_nb,
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
            self.trail,
            self.trail_top,
            self.pos,
            self.level_stk,
            self.stks_top,
            self.entailed_propagator_depths,
            self.entailment_trail,
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
            IDEMPOTENT,
            self.objective,
            self.decision,
            self.status,
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
        :param bound: the side of the variable's domain to tighten (MAX when minimizing, MIN when maximizing)
        :type bound: int
        :param mode: the optimization mode
        :type mode: str

        :return: whether the search can continue
        :rtype: bool
        """
        if mode == OPTIM_RESET:
            logger.debug("Resetting solver")
            self._cp_init()
            if not fix_choice_point(self.state, self.trail, self.trail_top, self.pos, variable, value, bound):
                return False
            buckets_init(self.triggered_propagators, self.problem.priorities)
        else:
            logger.debug("Pruning choice points")
            # arm the objective and let backtrack apply it: the bound is re-applied to each level the
            # search resumes, so the levels it kills are dropped as they are reached rather than up front
            self.objective[OBJ_VARIABLE] = variable
            self.objective[OBJ_BOUND] = bound
            self.objective[OBJ_VALUE] = value
            if not self._backtrack():
                return False
        return True

    def _cp_init(self) -> None:
        """
        Resets the search to the root.
        """
        cp_init(
            self.state,
            self.trail_top,
            self.pos,
            self.level_stk,
            self.stks_top,
            self.entailed_propagator_depths,
            self.entailment_trail,
            self.initial_domains,
            self.problem.unbound_variable_nb,
        )

    def _grow(self) -> None:
        """
        Doubles whichever caller-allocated array the search ran out of, and lets it continue.

        Nothing of the search is lost. The trail keeps its contents, so every mark and every position in
        pos still addresses the same entry; the level stack keeps its rows. Sizing either array for its
        worst case instead -- depth x (2 x domain_nb + 1) trail entries -- would hand back the memory
        this representation wins, and a hard failure would end a long optimization run for no reason.
        """
        if self.status[SOLVER_STATUS] == SOLVER_TRAIL_FULL:
            trail = np.empty((2 * len(self.trail), 2), dtype=np.int32)
            trail[: len(self.trail)] = self.trail
            self.trail = trail
            logger.info(f"The trail grew to {len(self.trail)} entries")
        else:
            level_stk = np.zeros((2 * len(self.level_stk), LEVEL_WIDTH), dtype=np.int32)
            level_stk[: len(self.level_stk)] = self.level_stk
            self.level_stk = level_stk
            logger.info(f"The stacks of the choice points grew to a maximal height of {len(self.level_stk)}")
        self.status[SOLVER_STATUS] = SOLVER_RUNNING

    def _backtrack(self) -> bool:
        """
        Backtracks by forwarding the solver state to the jitted backtrack.

        :return: true iff it was possible to backtrack
        :rtype: bool
        """
        return backtrack(
            self.statistics,
            self.state,
            self.trail,
            self.trail_top,
            self.pos,
            self.level_stk,
            self.stks_top,
            self.entailed_propagator_depths,
            self.entailment_trail,
            self.triggered_propagators,
            self.problem.triggers,
            self.problem.triggers_offsets,
            self.problem.priorities,
            self.problem.propagator_nb,
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

        Beyond the global counters, the dictionary breaks the two filtering counters down per propagator
        algorithm, restricted to the algorithms that ran at least once so that the breakdown stays readable.
        Each entry is suffixed with the algorithm name, so a breakdown sorts next to the total it partitions.

        A call that prunes nothing still costs a bucket pop, a gather of its variables' domains into the
        scratch buffer, an indirect call and a write-back, so a high no-change share on a given algorithm is
        where wasted propagation is concentrated.

        :return: a dictionary mapping statistic labels to values
        :rtype: Dict[str, int]
        """
        statistics = {
            STATS_LBL_ALG_BC_NB: int(self.statistics[STATS_IDX_ALG_BC_NB]),
            STATS_LBL_PROPAGATOR_ENTAILMENT_NB: int(self.statistics[STATS_IDX_PROPAGATOR_ENTAILMENT_NB]),
            STATS_LBL_PROPAGATOR_FILTER_NB: int(self.statistics[STATS_IDX_PROPAGATOR_FILTER_NB]),
            STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB: int(self.statistics[STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB]),
            STATS_LBL_PROPAGATOR_INCONSISTENCY_NB: int(self.statistics[STATS_IDX_PROPAGATOR_INCONSISTENCY_NB]),
            STATS_LBL_SOLVER_BACKTRACK_NB: int(self.statistics[STATS_IDX_SOLVER_BACKTRACK_NB]),
            STATS_LBL_SOLVER_CHOICE_NB: int(self.statistics[STATS_IDX_SOLVER_CHOICE_NB]),
            STATS_LBL_SOLVER_CHOICE_DEPTH: int(self.statistics[STATS_IDX_SOLVER_CHOICE_DEPTH]),
            STATS_LBL_SOLUTION_NB: int(self.statistics[STATS_IDX_SOLUTION_NB]),
            # the statistics array accumulates nanoseconds, the reported statistic is in milliseconds
            STATS_LBL_SOLVER_ELAPSED_TIME: int(self.statistics[STATS_IDX_SOLVER_ELAPSED_TIME]) // 1_000_000,
        }
        for algorithm in range(get_algorithm_nb()):
            base = STATS_MAX + STATS_ALG_WIDTH * algorithm
            calls = int(self.statistics[base + STATS_ALG_IDX_FILTER_NB])
            if calls:
                name = COMPUTE_DOMAINS_FCTS[algorithm].__name__.replace("compute_domains_", "").upper()
                statistics[f"{STATS_LBL_PROPAGATOR_FILTER_NB}_{name}"] = calls
                statistics[f"{STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB}_{name}"] = int(
                    self.statistics[base + STATS_ALG_IDX_FILTER_NO_CHANGE_NB]
                )
        return statistics


@njit(cache=True)
def solve_one(
    propagator_nb: int,
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
    trail: NDArray,
    trail_top: NDArray,
    pos: NDArray,
    level_stk: NDArray,
    stks_top: NDArray,
    entailed_propagator_depths: NDArray,
    entailment_trail: NDArray,
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
    idempotent: NDArray,
    objective: NDArray,
    decision: NDArray,
    status: NDArray,
    trail_headroom: int,
) -> NDArray | None:
    """
    Find at most one solution.

    Expects the propagation queue to already hold the propagators that need to run: the callers enqueue
    all the propagators (buckets_init) before the first call, and rely on backtrack to schedule the
    propagators affected by a refutation, or by the objective bound it re-applies, between subsequent calls.

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
    :param trail: the undo log of (flat index, old value) pairs
    :type trail: NDArray
    :param trail_top: the trail size as a Numpy array
    :type trail_top: NDArray
    :param pos: the index of the last trail entry per positionally guarded cell
    :type pos: NDArray
    :param level_stk: the per-level metadata
    :type level_stk: NDArray
    :param stks_top: the index of the top of the stacks as a Numpy array
    :type stks_top: NDArray
    :param entailed_propagator_depths: the depth at which each propagator was entailed, -1 when active
    :type entailed_propagator_depths: NDArray
    :param entailment_trail: the entailment trail, the first cell holds the trail size
    :type entailment_trail: NDArray
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
    :param idempotent: whether each algorithm reaches its own fixpoint in a single call, indexed by
                       algorithm rather than by propagator
    :type idempotent: NDArray
    :param objective: the objective as a Numpy array of variable, bound and value,
                      whose variable is -1 when not optimizing
    :type objective: NDArray
    :param decision: a scratch array the domain heuristic writes its split value into
    :type decision: NDArray
    :param status: set to SOLVER_TRAIL_FULL or SOLVER_LEVELS_FULL when the search stops for want of room
    :type status: NDArray
    :param trail_headroom: the trail entries any one step of the search can need
    :type trail_headroom: int

    :return: the solution if it exists or None
    :rtype: Optional[NDArray]
    """
    consistency_alg_fct = consistency_alg_fcts[0]
    nb_searches = len(decision_variables_offsets) - 1
    max_top = len(level_stk) - 3  # a ternary split pushes two levels and marks a third
    while True:
        # the arrays are caller-allocated, so the search stops for the solver to grow one rather than
        # overrun it silently -- with boundscheck off, the overrun is what would otherwise happen
        if trail_top[0] + trail_headroom > len(trail):
            status[SOLVER_STATUS] = SOLVER_TRAIL_FULL
            return None
        if stks_top[0] > max_top:
            status[SOLVER_STATUS] = SOLVER_LEVELS_FULL
            return None
        problem_status = consistency_alg_fct(
            propagator_nb,
            statistics,
            algorithms,
            priorities,
            offsets,
            propagator_variables,
            propagator_parameters,
            triggers,
            triggers_offsets,
            state,
            domains,
            trail,
            trail_top,
            pos,
            level_stk,
            stks_top,
            entailed_propagator_depths,
            entailment_trail,
            triggered_propagators,
            compute_domains_fcts,
            domain_buffer,
            idempotent,
        )
        top = stks_top[0]
        if problem_status == PROBLEM_BOUND:
            statistics[STATS_IDX_SOLUTION_NB] += 1
            return get_solution(domains)
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
                    # the heuristic only says where to split; branch owns the two levels it takes to do so
                    kind = dom_heuristic_fcts[search_idx](
                        domains,
                        variable,
                        dom_heuristic_params[
                            dom_heuristic_params_offsets[search_idx] : dom_heuristic_params_offsets[search_idx + 1]
                        ].reshape(
                            dom_heuristic_params_shapes[search_idx, 0], dom_heuristic_params_shapes[search_idx, 1]
                        ),
                        decision,
                    )
                    events = branch(
                        state,
                        trail,
                        trail_top,
                        pos,
                        level_stk,
                        stks_top,
                        variable,
                        kind,
                        decision[DECISION_VALUE],
                    )
                    top = stks_top[0]
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
                    statistics[STATS_IDX_SOLVER_CHOICE_NB] += 1
                    statistics[STATS_IDX_SOLVER_CHOICE_DEPTH] = max(statistics[STATS_IDX_SOLVER_CHOICE_DEPTH], top)
                    branched = True
                    break
        # either the problem is inconsistent, or no search can claim a variable although variables remain
        # unbound -- a level whose domains admit no assignment. Both are dead ends: backtrack.
        if not branched and not backtrack(
            statistics,
            state,
            trail,
            trail_top,
            pos,
            level_stk,
            stks_top,
            entailed_propagator_depths,
            entailment_trail,
            triggered_propagators,
            triggers,
            triggers_offsets,
            priorities,
            propagator_nb,
            objective,
        ):
            return None


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
        arity = np.int64(offsets[propagator_idx + 1, VARIABLE] - offsets[propagator_idx, VARIABLE])
        max_arity = max(max_arity, arity)
    return np.empty((max_arity, 2), dtype=np.int32)
