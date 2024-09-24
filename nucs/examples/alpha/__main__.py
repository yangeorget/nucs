from rich import print

from nucs.examples.alpha.alpha_problem import AlphaProblem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.solvers.heuristics import min_value_dom_heuristic, smallest_domain_var_heuristic
from nucs.statistics import get_statistics

if __name__ == "__main__":
    problem = AlphaProblem()
    solver = BacktrackSolver(
        problem, var_heuristic=smallest_domain_var_heuristic, dom_heuristic=min_value_dom_heuristic
    )
    solutions = solver.find_all()
    print(get_statistics(solver.statistics))
    print(problem.solution_as_dict(solutions[0]))
