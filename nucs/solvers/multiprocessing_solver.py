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

from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import get_statistics, init_statistics


class MultiprocessingSolver(Solver):
    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = init_statistics()
        self.solvers = solvers
        self.processes = []
        self.queue = Queue()

    def init_problem(self):
        for solver in self.solvers:
            solver.init_problem()

    def solve(self) -> Iterator[List[int]]:
        self.init_problem()
        for solver_idx, solver in enumerate(self.solvers):
            process = Process(target=solver.solve_one_interruptible, args=(solver_idx, self.queue))
            self.processes.append(process)
            process.start()
        while (solution := self.solve_one()) is not None:
            yield solution
            if not self.backtrack():
                break

    def solve_one(self) -> Optional[List[int]]:
        solution, statistics = self.queue.get()
        self.statistics = statistics
        return list(solution)

    def optimize(self, variable_idx: int) -> Optional[List[int]]:
        pass

    def reset(self) -> None:
        pass


if __name__ == "__main__":
    solvers = []
    for i in range(4):
        problem = QueensProblem(28)
        problem.shr_domains_lst[0] = (0 + i * 7, 6 + i * 7)
        solvers.append(BacktrackSolver(problem))
    solver = MultiprocessingSolver(solvers)
    solution = next(solver.solve())
    print(solution)
    print(get_statistics(solver.statistics))
