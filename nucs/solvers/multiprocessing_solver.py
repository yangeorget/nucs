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
import operator
from multiprocessing import Process, Queue
from typing import Callable, Iterator, List, Optional

from numpy.typing import NDArray

from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import init_statistics


class MultiprocessingSolver(Solver):
    """
    A solver relying on the multiprocessing package.
    This solver delegates resolution to a set of solvers.
    """

    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = [init_statistics() for _ in solvers]
        self.solvers = solvers

    def solve(self) -> Iterator[NDArray]:
        solution_queue: Queue = Queue()
        for proc_idx, solver in enumerate(self.solvers):
            Process(target=solver.solve_and_queue, args=(proc_idx, solution_queue)).start()
        nb = len(self.solvers)
        while nb > 0:
            proc_idx, solution, statistics = solution_queue.get()
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
        solution_queue: Queue = Queue()
        for proc_idx, solver in enumerate(self.solvers):
            Process(target=(getattr(solver, proc_func_name)), args=(variable_idx, proc_idx, solution_queue)).start()
        best_solution = None
        nb = len(self.solvers)
        while nb > 0:
            proc_idx, solution, statistics = solution_queue.get()
            self.statistics[proc_idx] = statistics
            if solution is None:
                nb -= 1
            elif best_solution is None or comparison_func(solution[variable_idx], best_solution[variable_idx]):
                best_solution = solution
        return best_solution
