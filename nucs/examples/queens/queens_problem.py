import argparse
from pprint import pprint

from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver
from nucs.statistics import get_statistics


class QueensProblem(Problem):
    """
    A simple model for the n-queens problem.

    CSPLIB problem #54 - https://www.csplib.org/Problems/prob054/
    """

    def __init__(self, n: int):
        super().__init__(
            [(0, n - 1)] * n,
            list(range(n)) * 3,
            [0] * n + list(range(n)) + list(range(0, -n, -1)),
        )
        self.add_propagator((list(range(n)), ALG_ALLDIFFERENT, []))
        self.add_propagator((list(range(n, 2 * n)), ALG_ALLDIFFERENT, []))
        self.add_propagator((list(range(2 * n, 3 * n)), ALG_ALLDIFFERENT, []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=12)
    args = parser.parse_args()
    problem = QueensProblem(args.n)
    solver = BacktrackSolver(problem)
    solver.solve_all()
    pprint(get_statistics(solver.statistics))
