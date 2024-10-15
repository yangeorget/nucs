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
from multiprocessing import Manager, Process
from typing import Iterator, List, Optional

from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import get_statistics, init_statistics


class MultiprocessingSolver(Solver):
    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = init_statistics()
        self.solvers = solvers

    def solve(self) -> Iterator[List[int]]:
        pass

    def solve_one(self) -> Optional[List[int]]:
        processes = []
        manager = Manager()
        solution = manager.list()
        run = manager.Event()
        run.set()
        for solver in self.solvers:
            process = Process(target=solver.solve_one, args=(run, solution))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        return list(solution)

    def minimize(self, variable_idx: int) -> Optional[List[int]]:
        pass

    def maximize(self, variable_idx: int) -> Optional[List[int]]:
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
    solver.solve_one()
    print(get_statistics(solver.statistics))
