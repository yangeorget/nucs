from nucs.problems.problem import Problem
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.propagators.propagators import ALG_ALLDIFFERENT

n = 8  # the number of queens
problem = Problem(
    [(0, n - 1)] * n,  # these n domains are shared between the 3n variables with different offsets
    list(range(n)) * 3,  # for each variable, its shared domain
    [0] * n + list(range(n)) + list(range(0, -n, -1))  # for each variable, its offset
)
problem.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
problem.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
problem.add_propagator((list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []))
print(BacktrackSolver(problem).find_one()[:n])