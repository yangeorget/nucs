import numpy as np

from ncs.constraints.sum import Sum
from ncs.problem import Problem
from ncs.solvers.simple_solver import SimpleSolver


class SumTest:
    def test_sum(self) -> None:
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
        problem.constraints.append(Sum(np.array([4, 0, 1, 2, 3])))
        solver = SimpleSolver(problem)
        solver.solve()
        print(domains)


if __name__ == "__main__":
    SumTest().test_sum()
