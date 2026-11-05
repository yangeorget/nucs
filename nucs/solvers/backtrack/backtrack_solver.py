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
from collections.abc import Sequence

from numpy.typing import NDArray

from nucs.constants import LOG_LEVEL_INFO
from nucs.numba_helper import build_function_ptrs
from nucs.problems.problem import Problem
from nucs.solvers.backtrack.backtrack_search_loop import run_search_loop_backtrack
from nucs.solvers.backtrack.consistency_algorithms import (
    CONSISTENCY_ALG_BC,
    CONSISTENCY_ALG_FCTS,
    SIGN_CONSISTENCY_ALG,
    get_consistency_algorithm_name,
)
from nucs.solvers.restart_policies import RESTART_POLICY_NEVER, RestartPolicy
from nucs.solvers.searches import SEARCHES_DEFAULT, Search
from nucs.solvers.tree_search_solver import TreeSearchSolver

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

    def _run_search_loop(self) -> tuple[int, NDArray | None]:
        """
        Runs the compiled search loop until it returns, by forwarding the solver state to the jitted run_search_loop_backtrack.

        :return: why the loop returned, and the solution when it found one
        :rtype: Tuple[int, Optional[NDArray]]
        """
        return run_search_loop_backtrack(
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
            self.bb,
            self.trail_headroom,
            self.interrupter.cells,
            self.propagator_weights,
            self.search_control,
            self.flat_searches.variable_searches,
        )
