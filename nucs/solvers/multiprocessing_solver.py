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
from typing import Iterator, List, Optional

from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.solver import Solver
from nucs.statistics import get_statistics, init_statistics


# TODO:
# interruptible solver ? could be a subclass of backtrack solver
# let caller create the subproblems or provide some help ?
# first make solve_one() works then solve()

class MultiprocessingSolver(Solver):
    def __init__(self, solvers: List[BacktrackSolver]):
        self.statistics = init_statistics()
        self.solvers = solvers

    def solve(self) -> Iterator[List[int]]:
        pass

    def solve_one(self) -> Optional[List[int]]:
        result = self.solvers[0].solve_one()
        self.statistics = self.solvers[0].statistics
        return result

    def minimize(self, variable_idx: int) -> Optional[List[int]]:
        pass

    def maximize(self, variable_idx: int) -> Optional[List[int]]:
        pass

    def reset(self) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    solvers = []
    for i in range(1):
        problem = QueensProblem(args.n)
        problem.shr_domains_lst[0] = (0, args.n-1)
        solvers.append(BacktrackSolver(problem))
    solver = MultiprocessingSolver(solvers)
    solver.solve_one()
    print(get_statistics(solver.statistics))
