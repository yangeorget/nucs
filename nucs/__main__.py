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
from timeit import default_timer as timer

from rich import print

from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver

if __name__ == "__main__":
    for i in range(8, 100):
        if i % 4 in [0, 1]:
            problem = BIBDProblem(i, (i * (i - 1)) // 4, i - 1, 4, 3)
            solver = BacktrackSolver(problem)
            start = timer()
            solver.solve_one()
            end = timer()
            print(f"{i} solve in {end - start} seconds")
