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
        self.queue: Queue = Queue()
        self.processes = [
            Process(target=solver.solve_queue, args=(idx, self.queue)) for idx, solver in enumerate(solvers)
        ]

    def solve(self) -> Iterator[List[int]]:
        for process in self.processes:
            process.start()
        nb = len(self.processes)
        while nb > 0:
            idx, solution, statistics = self.queue.get()
            self.statistics[idx] = statistics
            if solution is None:
                nb -= 1
            else:
                yield solution

    def optimize(self, variable_idx: int) -> Optional[List[int]]:
        # TODO: fix this
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-processors", type=int, default=1)
    args = parser.parse_args()
    problems = QueensProblem(12).split(args.processors, 0)
    solver = MultiprocessingSolver([BacktrackSolver(problem) for problem in problems])
    solver.solve_all()
    print(get_statistics(solver.statistics))
