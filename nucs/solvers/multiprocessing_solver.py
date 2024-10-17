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
import argparse
from multiprocessing import Process, Queue
from typing import Iterator, List, Optional

from rich import print

from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import get_statistics, init_statistics


class MultiprocessingSolver(Solver):
    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = [init_statistics() for _ in solvers]
        self.solvers = solvers
        self.queue: Queue = Queue()

    def solve(self) -> Iterator[List[int]]:
        for processor_idx, solver in enumerate(self.solvers):
            Process(target=solver.solve_queue, args=(processor_idx, self.queue)).start()
        nb = len(self.solvers)
        while nb > 0:
            processor_idx, solution, statistics = self.queue.get()
            self.statistics[processor_idx] = statistics
            if solution is None:
                nb -= 1
            else:
                yield solution

    def minimize(self, variable_idx: int) -> Optional[List[int]]:
        for processor_idx, solver in enumerate(self.solvers):
            Process(target=solver.minimize_queue, args=(variable_idx, processor_idx, self.queue)).start()
        solution = None
        nb = len(self.solvers)
        while nb > 0:
            processor_idx, new_solution, statistics = self.queue.get()
            self.statistics[processor_idx] = statistics
            if new_solution is None:
                nb -= 1
            elif solution is None or new_solution[variable_idx] < solution[variable_idx]:
                solution = new_solution
        return solution

    def maximize(self, variable_idx: int) -> Optional[List[int]]:
        for processor_idx, solver in enumerate(self.solvers):
            Process(target=solver.maximize_queue, args=(variable_idx, processor_idx, self.queue)).start()
        solution = None
        nb = len(self.solvers)
        while nb > 0:
            processor_idx, new_solution, statistics = self.queue.get()
            self.statistics[processor_idx] = statistics
            if new_solution is None:
                nb -= 1
            elif solution is None or new_solution[variable_idx] > solution[variable_idx]:
                solution = new_solution
        return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-processors", type=int, default=1)
    args = parser.parse_args()
    problems = QueensProblem(12).split(args.processors, 0)
    solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
    solver.solve_all()
    print(get_statistics(solver.statistics))
