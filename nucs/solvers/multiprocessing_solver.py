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
    # TODO: use shared memory instead of manager

    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = init_statistics()
        self.solvers = solvers

    def solve(self) -> Iterator[List[int]]:
        pass

    def solve_one(self) -> Optional[List[int]]:
        processes = []
        manager = Manager()
        out = manager.dict()
        out["solution"] = None
        run = manager.Event()
        run.set()
        for solver in self.solvers:
            solver.problem.init_problem(solver.statistics)
            process = Process(target=solver.solve_one_interruptible, args=(run, out))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        return list(out["solution"])

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
    solution = solver.solve_one()
    print(solution)
    print(get_statistics(solver.statistics))
