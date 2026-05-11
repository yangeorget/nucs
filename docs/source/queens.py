from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver

n = 8  # the number of queens
problem = Problem([(0, n - 1)] * n)
variables = range(0, n)
problem.add_propagator(ALG_ALLDIFFERENT, variables)
problem.add_propagator(ALG_ALLDIFFERENT, variables, range(n))
problem.add_propagator(ALG_ALLDIFFERENT, variables, range(0, -n, -1))
solver = BacktrackSolver(problem)
solution = next(solver.solve(), None)
print(solution[:n])
