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
# Copyright 2024 - Yan Georget
###############################################################################
import logging
import operator
from multiprocessing import Process, Queue
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
    STATS_IDX_SOLVER_BACKTRACK_NB,
    STATS_IDX_SOLVER_CHOICE_DEPTH,
    STATS_IDX_SOLVER_CHOICE_NB,
    STATS_IDX_SOLVER_SOLUTION_NB,
    STATS_LBL_ALG_BC_NB,
    STATS_LBL_ALG_BC_WITH_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_CHANGE_NB,
    STATS_LBL_ALG_SHAVING_NB,
    STATS_LBL_ALG_SHAVING_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_ENTAILMENT_NB,
    STATS_LBL_PROPAGATOR_FILTER_NB,
    STATS_LBL_PROPAGATOR_FILTER_NO_CHANGE_NB,
    STATS_LBL_PROPAGATOR_INCONSISTENCY_NB,
    STATS_LBL_SOLVER_BACKTRACK_NB,
    STATS_LBL_SOLVER_CHOICE_DEPTH,
    STATS_LBL_SOLVER_CHOICE_NB,
    STATS_LBL_SOLVER_SOLUTION_NB,
)
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver

logger = logging.getLogger(__name__)


class MultiprocessingSolver(Solver):
    """
    A solver relying on the multiprocessing package.
    This solver delegates resolution to a set of solvers.
    """

    def __init__(self, solvers: List[BacktrackSolver], log_level: str = LOG_LEVEL_INFO):
        super().__init__(None, log_level)
        logger.info(f"MultiprocessingSolver has {len(solvers)} processors")
        self.solvers = solvers
        logger.debug("Initializing statistics")
        self.statistics = [None for _ in solvers]
        logger.debug("Statistics initialized")
        logger.debug("MultiprocessingSolver initialized")

    def get_statistics(self) -> Dict[str, int]:
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
            STATS_LBL_SOLVER_SOLUTION_NB: sum_stats(self.statistics, STATS_IDX_SOLVER_SOLUTION_NB),
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

    def minimize(self, variable_idx: int) -> Optional[NDArray]:
        return self.optimize(variable_idx, "minimize_and_queue", operator.lt)

    def maximize(self, variable_idx: int) -> Optional[NDArray]:
        return self.optimize(variable_idx, "maximize_and_queue", operator.gt)

    def optimize(self, variable_idx: int, proc_func_name: str, comparison_func: Callable) -> Optional[NDArray]:
        solutions: Queue = Queue()
        for proc_idx, solver in enumerate(self.solvers):
            Process(target=(getattr(solver, proc_func_name)), args=(variable_idx, proc_idx, solutions)).start()
        best_solution = None
        nb = len(self.solvers)
        while nb > 0:
            proc_idx, solution, statistics = solutions.get()
            self.statistics[proc_idx] = statistics
            if solution is None:
                nb -= 1
            elif best_solution is None or comparison_func(solution[variable_idx], best_solution[variable_idx]):
                best_solution = solution
        return best_solution


def sum_stats(stats: List[Any], index: int) -> int:
    return sum(int(s[index]) for s in stats)


def max_stats(stats: List[Any], index: int) -> int:
    return max(int(s[index]) for s in stats)
