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
from typing import Callable, Iterator, List, Optional

from numpy.typing import NDArray

from nucs.constants import LOG_FORMAT
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import init_statistics

logger = logging.getLogger(__name__)


class MultiprocessingSolver(Solver):
    """
    A solver relying on the multiprocessing package.
    This solver delegates resolution to a set of solvers.
    """

    def __init__(self, solvers: List[BacktrackSolver], log_level: int = logging.INFO):
        logging.basicConfig(format=LOG_FORMAT, level=log_level)
        logger.info(f"Initializing MultiprocessingSolver with {len(solvers)} processors")
        super().__init__(None)
        self.solvers = solvers
        logger.info("Initializing statistics")
        self.statistics = [init_statistics() for _ in solvers]
        logger.info("Statistics initialized")
        logger.info("MultiprocessingSolver initialized")

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
