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
from multiprocessing import Process, Queue
from typing import Iterator, List, Optional

from numpy.typing import NDArray

from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import init_statistics


class MultiprocessingSolver(Solver):
    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = [init_statistics() for _ in solvers]
        self.solvers = solvers

    def solve(self) -> Iterator[NDArray]:
        queue: Queue = Queue()
        for processor_idx, solver in enumerate(self.solvers):
            Process(target=solver.solve_and_queue, args=(processor_idx, queue)).start()
        nb = len(self.solvers)
        while nb > 0:
            processor_idx, solutions, statistics = queue.get()
            self.statistics[processor_idx] = statistics
            for solution in solutions:
                if solution is None:
                    nb -= 1
                else:
                    yield solution

    def minimize(self, variable_idx: int) -> Optional[NDArray]:
        queue: Queue = Queue()
        for processor_idx, solver in enumerate(self.solvers):
            Process(target=solver.minimize_and_queue, args=(variable_idx, processor_idx, queue)).start()
        best_solution = None
        nb = len(self.solvers)
        while nb > 0:
            processor_idx, solutions, statistics = queue.get()
            self.statistics[processor_idx] = statistics
            for solution in solutions:
                if solution is None:
                    nb -= 1
                elif best_solution is None or solution[variable_idx] < best_solution[variable_idx]:
                    best_solution = solution
        return best_solution

    def maximize(self, variable_idx: int) -> Optional[NDArray]:
        queue: Queue = Queue()
        for processor_idx, solver in enumerate(self.solvers):
            Process(target=solver.maximize_and_queue, args=(variable_idx, processor_idx, queue)).start()
        best_solution = None
        nb = len(self.solvers)
        while nb > 0:
            processor_idx, solutions, statistics = queue.get()
            self.statistics[processor_idx] = statistics
            for solution in solutions:
                if solution is None:
                    nb -= 1
                elif best_solution is None or solution[variable_idx] > best_solution[variable_idx]:
                    best_solution = solution
        return best_solution
