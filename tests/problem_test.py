import numpy as np

from ncs.problem import Problem
from ncs.propagators.sum import Sum


class TestProblem:
    def test_filter(self) -> None:
        domains = np.array([[0, 2], [0, 2], [4, 6]])
        problem = Problem(domains)
        problem.propagators.append(Sum(np.array([2, 0, 1])))
        assert problem.filter(None)
        assert problem.is_solved()
        assert np.all(problem.domains == np.array([[2, 2], [2, 2], [4, 4]]))

    def test_update_domaines(self) -> None:
        # TODO: test changes
        pass
