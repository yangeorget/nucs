import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.sum import Sum


class TestProblem:
    def test_filter(self) -> None:
        domains = np.array([[0, 2], [0, 2], [4, 6]])
        problem = Problem(domains, [Sum([2, 0, 1])])
        assert problem.filter(None)
        assert not problem.is_not_solved()
        assert np.all(problem.domains == np.array([[2, 2], [2, 2], [4, 4]]))

    def test_update_domains(self) -> None:
        domains = np.array([[0, 2], [0, 2], [4, 6]])
        problem = Problem(domains)
        changes = np.zeros((3, 2), dtype=bool)
        propagator = Sum([2, 0, 1])
        problem.update_domains(propagator, changes)
        assert changes[0][0]
        assert not changes[0][1]
        assert changes[1][0]
        assert not changes[1][1]
        assert not changes[2][0]
        assert changes[2][1]
        assert np.all(problem.domains == np.array([[2, 2], [2, 2], [4, 4]]))
