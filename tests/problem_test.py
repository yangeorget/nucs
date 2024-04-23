import numpy as np

from ncs.constraints.sum import Sum
from ncs.problem import Problem


class TestProblem:
    def test_sum(self) -> None:
        domains = np.array(
            [
                [0, 2],
                [0, 2],
                [4, 6],
            ]
        )
        problem = Problem(domains)
        problem.constraints.append(Sum(np.array([2, 0, 1])))
        assert problem.filter()
        assert problem.is_solved()
        assert np.all(problem.domains == np.array([[2, 2], [2, 2], [4, 4]]))
