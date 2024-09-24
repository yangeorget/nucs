from rich import print

from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics

if __name__ == "__main__":
    problem = BIBDProblem(8, 14, 7, 4, 3)
    solver = BacktrackSolver(problem)
    solver.solve_all()
    print(get_statistics(solver.statistics))
