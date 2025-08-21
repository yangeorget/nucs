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
# Copyright 2024-2025 - Yan Georget
###############################################################################
import logging
import operator
from multiprocessing import Process, Queue, set_start_method
from typing import Any, Callable, Dict, Iterator, List, Optional

from numpy.typing import NDArray

from nucs.constants import (
    LOG_LEVEL_INFO,
    STATS_IDX_ALG_BC_NB,
    STATS_IDX_ALG_BC_WITH_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_CHANGE_NB,
    STATS_IDX_ALG_SHAVING_NB,
    STATS_IDX_ALG_SHAVING_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_ENTAILMENT_NB,
    STATS_IDX_PROPAGATOR_FILTER_NB,
    STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_IDX_PROPAGATOR_INCONSISTENCY_NB,
    STATS_IDX_SOLUTION_NB,
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_LBL_ALG_BC_NB,
    STATS_LBL_ALG_BC_WITH_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_CHANGE_NB,
    STATS_LBL_ALG_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_ENTAILMENT_NB,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_INCONSISTENCY_NB,
    STATS_LBL_SOLUTION_NB,
    STATS_LBL_SOLVER_BACKTRACK_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
    STATS_LBL_SOLVER_CHOICE_NB,
)
from nucs.solvers.queue_solver import QueueSolver
from nucs.solvers.solver import Solver

logger = logging.getLogger(__name__)


class MultiprocessingSolver(Solver):
    """
    A solver relying on the multiprocessing package.
    This solver delegates resolution to a set of solvers.
    """

    def __init__(self, solvers: List[QueueSolver], log_level: str = LOG_LEVEL_INFO):
        super().__init__(None, log_level)
        logger.info(f"MultiprocessingSolver has {len(solvers)} processors")
        set_start_method("fork", force=True)
        self.solvers = solvers
        logger.debug("Initializing statistics")
        self.statistics = [solver.get_statistics_as_array() for solver in solvers]
        logger.debug("Statistics initialized")
        logger.debug("MultiprocessingSolver initialized")

    def get_statistics_as_dictionary(self) -> Dict[str, int]:
        return {
            STATS_LBL_ALG_BC_NB: sum_stats(self.statistics, STATS_IDX_ALG_BC_NB),
            STATS_LBL_ALG_BC_WITH_SHAVING_NB: sum_stats(self.statistics, STATS_IDX_ALG_BC_WITH_SHAVING_NB),
            STATS_LBL_ALG_SHAVING_NB: sum_stats(self.statistics, STATS_IDX_ALG_SHAVING_NB),
            STATS_LBL_ALG_SHAVING_CHANGE_NB: sum_stats(self.statistics, STATS_IDX_ALG_SHAVING_CHANGE_NB),
            STATS_LBL_ALG_SHAVING_NO_CHANGE_NB: sum_stats(self.statistics, STATS_IDX_ALG_SHAVING_NO_CHANGE_NB),
            STATS_LBL_PROPAGATOR_ENTAILMENT_NB: sum_stats(self.statistics, STATS_IDX_PROPAGATOR_ENTAILMENT_NB),
            STATS_LBL_PROPAGATOR_FILTER_NB: sum_stats(self.statistics, STATS_IDX_PROPAGATOR_FILTER_NB),
            STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB: sum_stats(
                self.statistics, STATS_IDX_PROPAGATOR_FILTER_NO_CHANGE_NB
            ),
            STATS_LBL_PROPAGATOR_INCONSISTENCY_NB: sum_stats(self.statistics, STATS_IDX_PROPAGATOR_INCONSISTENCY_NB),
            STATS_LBL_SOLVER_BACKTRACK_NB: sum_stats(self.statistics, STATS_IDX_SOLVER_BACKTRACK_NB),
            STATS_LBL_SOLVER_CHOICE_NB: sum_stats(self.statistics, STATS_IDX_SOLVER_CHOICE_NB),
            STATS_LBL_SOLVER_CHOICE_DEPTH: max_stats(self.statistics, STATS_IDX_SOLVER_CHOICE_DEPTH),
            STATS_LBL_SOLUTION_NB: sum_stats(self.statistics, STATS_IDX_SOLUTION_NB),
        }

    def solve(self) -> Iterator[NDArray]:
        solutions: Queue = Queue()
        for proc_idx, solver in enumerate(self.solvers):
            Process(target=solver.solve_and_queue, args=(proc_idx, solutions)).start()
        nb = len(self.solvers)
        while nb > 0:
            proc_idx, solution, statistics = solutions.get()
            self.statistics[proc_idx] = statistics
            if solution is None:
                nb -= 1
            else:
                yield solution

    def minimize(self, variable: int, mode: str) -> Optional[NDArray]:
        return self.optimize(variable, "minimize_and_queue", operator.lt, mode)

    def maximize(self, variable: int, mode: str) -> Optional[NDArray]:
        return self.optimize(variable, "maximize_and_queue", operator.gt, mode)

    def optimize(self, variable: int, proc_func_name: str, comparison_func: Callable, mode: str) -> Optional[NDArray]:
        solutions: Queue = Queue()
        for proc_idx, solver in enumerate(self.solvers):
            Process(target=(getattr(solver, proc_func_name)), args=(variable, proc_idx, solutions, mode)).start()
        best_solution = None
        nb = len(self.solvers)
        while nb > 0:
            proc_idx, solution, statistics = solutions.get()
            self.statistics[proc_idx] = statistics
            if solution is None:
                nb -= 1
            elif best_solution is None or comparison_func(solution[variable], best_solution[variable]):
                best_solution = solution
        return best_solution


def sum_stats(stats: List[Any], index: int) -> int:
    return sum(int(s[index]) for s in stats if s is not None)


def mean_stats(stats: List[Any], index: int) -> int:
    return sum_stats(stats, index) // len(stats)


def max_stats(stats: List[Any], index: int) -> int:
    return max(int(s[index]) for s in stats if s is not None)
