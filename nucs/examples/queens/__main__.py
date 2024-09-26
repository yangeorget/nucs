import argparse

from rich import print

from nucs.examples.queens.queens_problem import QueensProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    problem = QueensProblem(args.n)
    solver = BacktrackSolver(problem)
    solver.solve_all()
    print(get_statistics(solver.statistics))
