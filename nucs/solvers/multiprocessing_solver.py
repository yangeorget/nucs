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

from rich import print

from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import get_statistics


class MultiprocessingSolver(Solver):
    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = list(range(len(solvers)))
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
    solvers = []
    queen_nb = 12
    processor_nb = 6
    size = queen_nb // processor_nb  # TODO: simplify this
    for i in range(processor_nb):
        problem = QueensProblem(queen_nb)
        problem.shr_domains_lst[0] = (0 + i * size, size - 1 + i * size)
        solvers.append(BacktrackSolver(problem))
    solver = MultiprocessingSolver(solvers)
    solver.solve_all()
    print(get_statistics(solver.statistics))
