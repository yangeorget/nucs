import argparse

from rich import print

from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int)
    parser.add_argument("-b", type=int)
    parser.add_argument("-r", type=int)
    parser.add_argument("-k", type=int)
    parser.add_argument("-l", type=int)
    args = parser.parse_args()
    problem = BIBDProblem(args.v, args.b, args.r, args.k, args.l)
    solver = BacktrackSolver(problem)
    solver.solve_all(lambda solution: print(problem.solution_as_matrix(solution)))
    print(get_statistics(solver.statistics))
