from rich import print

from nucs.examples.bibd.bibd_problem import BIBDProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics

if __name__ == "__main__":
    problem = BIBDProblem(8, 14, 7, 4, 3)
    # problem = BIBDProblem(6, 10, 5, 3, 2)
    # problem = BIBDProblem(7, 7, 3, 3, 1)
    solver = BacktrackSolver(problem)
    solver.solve_all(lambda solution: print(problem.solution_as_matrix(solution)))
    print(get_statistics(solver.statistics))