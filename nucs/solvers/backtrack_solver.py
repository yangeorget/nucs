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
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from nucs.buckets import buckets_create, buckets_init
from nucs.constants import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    LOG_LEVEL_INFO,
    OBJECTIVE_BOUND,
    OBJECTIVE_VALUE,
    OBJECTIVE_VARIABLE,
    OBJECTIVE_WIDTH,
)
from nucs.heuristics.heuristics import DOM_HEURISTIC_FCTS, SIGN_DOM_HEURISTIC, SIGN_VAR_HEURISTIC, VAR_HEURISTIC_FCTS
from nucs.numba_helper import (
    NUMBA_DISABLE_JIT,
    addresses_from_functions,
    build_function_ptrs,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import (
    ALG_DUMMY,
    COMPUTE_DOMAINS_FCTS,
    SIGN_COMPUTE_DOMAINS,
    get_algorithm_names,
    get_algorithm_nb,
)
from nucs.solvers.choice_points import CHOICE_POINT_WIDTH, backtrack, choice_point_init, tighten_objective
from nucs.solvers.consistency_algorithms import (
    CONSISTENCY_ALG_BC,
    CONSISTENCY_ALG_FCTS,
    SIGN_CONSISTENCY_ALG,
    get_consistency_algorithm_name,
)
from nucs.solvers.interruption import INTERRUPTION_DEADLINE, Interruption
from nucs.solvers.restarts import NO_RESTARTS, Restarts
from nucs.solvers.search import DEFAULT_SEARCHES, Search, flatten_searches
from nucs.solvers.search_arrays import allocate_search_arrays, get_domain_buffer
from nucs.solvers.search_loop import (
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
    solve_one_step,
)
from nucs.solvers.solver import OPTIM_RESET, Solver, format_timeout
from nucs.solvers.weights import weights_init
from nucs.statistics import (
    STATS_IDX_SOLVER_ELAPSED_TIME,
    STATS_IDX_SOLVER_RESTART_NB,
    statistics_as_dictionary,
    statistics_init,
)

logger = logging.getLogger(__name__)


class BacktrackSolver(Solver):
    """
    A solver relying on a backtracking mechanism.
    """

    # the failure limits of the descents; each search draws its own, from the first one (see _iterate_solutions)
    restart_limits: Iterator[int]

    def __init__(
        self,
        problem: Problem,
        consistency_algorithm: int = CONSISTENCY_ALG_BC,
        searches: Sequence[Search] = DEFAULT_SEARCHES,
        log_level: str = LOG_LEVEL_INFO,
        weight_decay: float = 1.0,
        restarts: Restarts = NO_RESTARTS,
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
                         DEFAULT_SEARCHES, one search that branches on every variable with the default heuristics of
                         Search. The union of the searches' decision variables should cover every branchable variable.
        :type searches: Sequence[Search]
        :param log_level: the log level,
                          defaults to INFO
        :type log_level: str
        :param weight_decay: in (0, 1], how much the failure weight of a propagator keeps of a failure after each
                             later failure; 1 counts all the failures the same (dom/wdeg), less than 1 prefers the
                             recent ones (AFC), defaults to 1
        :type weight_decay: float
        :param restarts: after how many failures each descent restarts from the root (nucs.solvers.restarts),
                         defaults to no restart. When enumerating, the restarts stop at the first solution, so that no
                         solution is found twice.
        :type restarts: Restarts
        :param last_conflict: whether the search branches first on the variable of the last refuted decision, as long
                              as it is unbound (last-conflict reasoning), defaults to False
        :type last_conflict: bool
        """
        super().__init__(problem, log_level)
        # every search keeps its own decision variables, variable and domain heuristics and their parameters
        self.flat_searches = flatten_searches(searches, problem.domain_nb)
        for description in self.flat_searches.searches_as_printable():
            logger.info(f"BacktrackSolver uses {description}")
        logger.info(
            f"BacktrackSolver uses consistency algorithm {get_consistency_algorithm_name(consistency_algorithm)}"
        )
        logger.info(f"BacktrackSolver uses {restarts}")
        logger.info(f"BacktrackSolver uses last-conflict: {'yes' if last_conflict else 'no'}")
        logger.info(f"BacktrackSolver uses a weight decay of {weight_decay}")
        self.triggered_propagators = buckets_create(problem.propagator_nb)
        self.domain_buffer = get_domain_buffer(problem.offsets)
        logger.debug("Initializing choice points")
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
        # the branch-and-bound bound, as OBJECTIVE_VARIABLE, OBJECTIVE_BOUND and OBJECTIVE_VALUE: the variable
        # optimized, the side of its domain to tighten, and the best value found so far. It is solver
        # state, not choice-point state -- the bound holds for the whole remaining search, so backtrack
        # re-applies it to each choice point it resumes rather than it being written into them all when it
        # is found, and nothing about it is trailed. A return to the root applies it there too, with a mark of 0.
        # OBJECTIVE_VARIABLE stays -1 until the first solution of an optimization, which is how backtrack and
        # _reset_to_root know there is no bound to apply. In OPTIM_RESET, the root already holds the bound, so
        # backtrack finds it applied and changes nothing.
        self.objective = np.full(OBJECTIVE_WIDTH, -1, dtype=np.int32)
        # read by solve_one_step at every node, and written by interrupt() and by the deadline of a search
        self.interruption = Interruption()
        self.restarts = restarts
        # the restart limit is drawn by each search, the other cells are reset with the root by _choice_point_init
        self.search_control = np.full(SEARCH_CONTROL_WIDTH, -1, dtype=np.int64)
        self.search_control[SEARCH_CONTROL_LAST_CONFLICT] = int(last_conflict)
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
        logger.debug("Statistics initialized")
        # the compiled compute_domains address of each algorithm, which the consistency algorithm calls a propagator
        # through; resolving only the algorithms used by the problem keeps the init cost proportional to the problem
        # instead of the whole propagator library; without the JIT this is a placeholder that call_compute_domains
        # ignores
        self.compute_domains_addrs = addresses_from_functions(
            COMPUTE_DOMAINS_FCTS, SIGN_COMPUTE_DOMAINS, np.unique(self.problem.algorithms), ALG_DUMMY
        )
        # the function tables threaded into solve_one_step: Numba typed lists under the JIT, plain Python lists
        # otherwise
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
        return self._backtrack()

    def optimize(self, variable: int, bound: int, mode: str, timeout: float | None = None) -> Iterator[NDArray]:
        logger.info(
            f"Optimizing and iterating over the solutions: {'minimizing' if bound == DOMAIN_MIN else 'maximizing'} "
            f"variable {variable} in {mode} mode, {format_timeout(timeout)}"
        )
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
        self.interrupted = False
        disarm = self.interruption.arm_deadline(timeout)
        try:
            t0 = time.perf_counter_ns()
            # no solution yet, so the search has no objective bound; _advance_after_optimum arms it.
            # An enumeration disarms it here too: the solver may have been optimized with before.
            self.objective[OBJECTIVE_VARIABLE] = -1
            self.restart_limits = self.restarts.limits()
            self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = next(self.restart_limits)
            # a previous search may have left the root tightened by its objective, or wiped out: start from the
            # initial domains, keeping what the propagator weights learned
            self._reset_to_root()
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
            if status == SOLVER_TRAIL_FULL:
                self._grow_trail()
            elif status == SOLVER_CHOICE_POINTS_FULL:
                self._grow_choice_points()
            elif status == SOLVER_RESTART:
                if not self._restart():
                    return None
            elif status == SOLVER_INTERRUPTED:
                # a timeout and interrupt() both cut the search: what it found is not proven
                self.interrupted = True
                if self.interruption.cell[0] == INTERRUPTION_DEADLINE:
                    logger.info("Timeout reached, stopping the search")
                else:
                    logger.info("Interrupted, stopping the search")
                return None
            else:
                raise AssertionError(f"Unknown solver status {status}")

    def _restart(self) -> bool:
        """
        Restarts the search from the root, with the next failure limit of the restart policy.

        :return: false when the root cannot hold a better solution, which proves the best solution so far optimal
        :rtype: bool
        """
        logger.debug("Restarting")
        self.statistics[STATS_IDX_SOLVER_RESTART_NB] += 1
        self.search_control[SEARCH_CONTROL_RESTART_LIMIT] = next(self.restart_limits)
        return self._reset_to_root()

    def _reset_to_root(self) -> bool:
        """
        Resets the search to the root and re-applies the best solution so far there.

        What the search learned stays: the propagator weights are not reset. The objective is tightened with a mark
        of 0, since the reset undoes every tightening above it: what this writes is undone only by the next reset.

        :return: false when the root cannot hold a better solution, which proves the best solution so far optimal
        :rtype: bool
        """
        self._choice_point_init()
        variable = self.objective[OBJECTIVE_VARIABLE]
        if variable >= 0:
            value = self.objective[OBJECTIVE_VALUE]
            bound = self.objective[OBJECTIVE_BOUND]
            if (
                tighten_objective(
                    self.state, self.trail_log, self.trail_top, self.trail_indices, 0, variable, value, bound
                )
                < 0
            ):
                return False
        # the search may have stopped at a failure: the failed filtering left propagators in the queue, and backtrack
        # added more, which buckets_init removes before it adds every propagator
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
        :attr:`interrupted` is set. The interruption is final, every later search stops at once too.

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
        # arm the objective in both modes: OPTIM_RESET applies it at the root, and OPTIM_PRUNE lets backtrack
        # apply it to each choice point the search resumes, so the choice points it kills are dropped as they are
        # reached rather than up front
        self.objective[OBJECTIVE_VARIABLE] = variable
        self.objective[OBJECTIVE_BOUND] = bound
        self.objective[OBJECTIVE_VALUE] = value
        if mode == OPTIM_RESET:
            logger.debug("Resetting solver")
            return self._reset_to_root()
        logger.debug("Pruning choice points")
        return self._backtrack()

    def _choice_point_init(self) -> None:
        """
        Resets the search to the root: the choice points, and the search control that counts from the root --
        the failures of the descent, and the conflict variable, which is forgotten.
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

    def _grow_trail(self) -> None:
        """
        Doubles the trail, which the search ran out of, and lets the search continue.

        Nothing of the search is lost. The trail keeps its contents, so every mark and every position in
        trail_indices still addresses the same entry. Sizing the trail for its worst case instead -- depth x
        (2 x domain_nb + 1) entries -- would hand back the memory this representation wins, and a hard failure
        would end a long optimization run for no reason.
        """
        trail = np.empty((2 * len(self.trail_log), 2), dtype=np.int32)
        trail[: len(self.trail_log)] = self.trail_log
        self.trail_log = trail
        logger.info(f"The trail grew to {len(self.trail_log)} entries")

    def _grow_choice_points(self) -> None:
        """
        Doubles the stack of choice points, which the search ran out of, and lets the search continue.

        Nothing of the search is lost: the stack keeps its rows.
        """
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
