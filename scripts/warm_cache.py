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
Warms the Numba JIT cache for all of NuCS's @njit code.

Run once with NUMBA_CACHE_DIR pointing at a persistent directory (e.g. at Docker build time) so that every
propagator, every variable/value heuristic and the consistency algorithm are compiled and cached ahead of
time, and no JIT happens at solve time.

The solver does not call these functions directly: it compiles each one for a fixed explicit signature via
``_get_wrapper_address`` (so it can dispatch through a function pointer). Numba keys cached artifacts by
function *and* signature, so the warm-up MUST compile with those same explicit signatures -- calling the
functions with example arrays would infer a different (e.g. C-contiguous) signature and the cache would miss
at run time. We therefore compile each function exactly as :class:`BacktrackSolver` does, then run a few tiny
solves to compile the remaining glue (the typed-list builders and inner-loop helpers).
"""

from nucs.constants import (
    SIGN_COMPUTE_DOMAINS,
    SIGN_CONSISTENCY_ALG,
    SIGN_DOM_HEURISTIC,
    SIGN_GET_TRIGGERS,
    SIGN_VAR_HEURISTIC,
)
from nucs.heuristics.heuristics import DOM_HEURISTIC_FCTS, VAR_HEURISTIC_FCTS
from nucs.numba_helper import addresses_from_functions
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT, COMPUTE_DOMAINS_FCTS, GET_TRIGGERS_FCTS
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.consistency_algorithms import CONSISTENCY_ALG_FCTS


def warm() -> None:
    """
    Compiles and caches every propagator, heuristic and consistency algorithm with the solver's signatures.
    """
    # Compile each function for the exact signature the solver dispatches through (see BacktrackSolver.__init__).
    addresses_from_functions(COMPUTE_DOMAINS_FCTS, SIGN_COMPUTE_DOMAINS)
    addresses_from_functions(GET_TRIGGERS_FCTS, SIGN_GET_TRIGGERS)
    addresses_from_functions(VAR_HEURISTIC_FCTS, SIGN_VAR_HEURISTIC)
    addresses_from_functions(DOM_HEURISTIC_FCTS, SIGN_DOM_HEURISTIC)
    addresses_from_functions(CONSISTENCY_ALG_FCTS, SIGN_CONSISTENCY_ALG)
    # A tiny solve compiles the remaining @njit glue (typed-list builders, choice-point and bucket helpers)
    # exactly as it is compiled at run time.
    problem = Problem([(0, 3)] * 4)
    problem.add_propagator(ALG_ALLDIFFERENT, range(4))
    next(BacktrackSolver(problem, log_level="ERROR").solve(), None)


if __name__ == "__main__":
    warm()
