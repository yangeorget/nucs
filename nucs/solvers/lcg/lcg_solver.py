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
The solver of lazy clause generation (design/proposals/lcg.md).
"""

from numpy.typing import NDArray

from nucs.solvers.lcg.lcg_search_loop import run_search_loop_lcg
from nucs.solvers.tree_search_solver import TreeSearchSolver


class LcgSolver(TreeSearchSolver):
    """
    A solver that will learn clauses from its failures (lazy clause generation).

    Its propagation is always bound consistency, and no parameter selects another one: the conflict analysis needs a
    reason for each change of a bound, and a custom consistency algorithm gives none.

    At this step (design/proposals/lcg.md, 7.0, step 4), it searches exactly as BacktrackSolver does, with its own
    compiled search loop and propagation loop, which are copies of those of BacktrackSolver: every counter is the same.
    The next steps change these copies and this class. BacktrackSolver and its loops do not change.
    """

    def _run_search_loop(self) -> tuple[int, NDArray | None]:
        """
        Runs the compiled search loop of LcgSolver until it returns, by forwarding the solver state to it.

        :return: why the loop returned, and the solution when it found one
        :rtype: Tuple[int, Optional[NDArray]]
        """
        return run_search_loop_lcg(
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
        )
