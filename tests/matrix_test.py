import numpy as np

from ncs.constraints.sum import Sum
from ncs.problem import Problem
from ncs.solvers.simple_solver import SimpleSolver


class MatrixTest:
    def test_matrix(self) -> None:
        domains = np.array(
            [
                [0, 2],
                [0, 2],
                [0, 2],
                [0, 2],
                [8, 12],
            ]
        )
        problem = Problem(domains)
        problem.add_constraint(Sum(np.array([4, 0, 1, 2, 3])))
        solver = SimpleSolver(problem)
        solver.solve()
        print(domains)


if __name__ == "__main__":
    MatrixTest().test_matrix()
