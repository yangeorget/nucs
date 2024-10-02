import argparse

from rich import print

from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import max_value_dom_heuristic
from nucs.statistics import get_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int)
    parser.add_argument("-b", type=int)
    parser.add_argument("-r", type=int)
    parser.add_argument("-k", type=int)
    parser.add_argument("-l", type=int)
    parser.add_argument("--symmetry_breaking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    problem = BIBDProblem(args.v, args.b, args.r, args.k, args.l, args.symmetry_breaking)
    solver = BacktrackSolver(problem, dom_heuristic=max_value_dom_heuristic)
    solver.solve_one()
    print(get_statistics(solver.statistics))
