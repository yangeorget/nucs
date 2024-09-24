from rich import print

from nucs.examples.donald.donald_problem import DonaldProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import get_statistics

if __name__ == "__main__":
    problem = DonaldProblem()
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    print(get_statistics(solver.statistics))
    for solution in solver.solve():
        print(problem.solution_as_dict(solution))
