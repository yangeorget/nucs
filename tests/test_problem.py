import numpy as np

from ncs.problems.problem import Problem
from ncs.propagators.sum import Sum


class TestProblem:
    def test_var_domains(self) -> None:
        shr_domains = np.array([[0, 2], [4, 6]])
        dom_indices = [0, 0, 1]
        dom_offsets = [0, 1, 2]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert np.all(problem.get_var_domains() == np.array([[0, 2], [1, 3], [6, 8]]))

    def test_lcl_domains(self) -> None:
        shr_domains = np.array([[0, 2], [4, 6]])
        dom_indices = [0, 0, 1]
        dom_offsets = [0, 1, 2]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        variables = np.array([0, 2])
        assert np.all(problem.get_lcl_domains(variables) == np.array([[0, 2], [6, 8]]))

    def test_set_lcl_mins(self) -> None:
        shr_domains = np.array([[0, 2], [4, 6]])
        dom_indices = [0, 0, 1]
        dom_offsets = [0, 1, 2]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        variables = np.array([0, 2])
        lcl_mins = np.array([1, 7])
        problem.set_lcl_mins(variables, lcl_mins)
        assert np.all(problem.shr_domains == np.array([[1, 2], [5, 6]]))

    def test_set_lcl_maxs(self) -> None:
        shr_domains = np.array([[0, 2], [4, 6]])
        dom_indices = [0, 0, 1]
        dom_offsets = [0, 1, 2]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        variables = np.array([0, 2])
        lcl_maxs = np.array([1, 7])
        problem.set_lcl_maxs(variables, lcl_maxs)
        assert np.all(problem.shr_domains == np.array([[0, 1], [4, 5]]))

    def test_is_not_instantiated(self) -> None:
        shr_domains = np.array([[0, 2], [0, 0]])
        dom_indices = [0, 0, 1]
        dom_offsets = [0, 1, 2]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert problem.is_not_instantiated(0)
        assert problem.is_not_instantiated(1)
        assert not problem.is_not_instantiated(2)

    def test_is_inconsistent_ko(self) -> None:
        shr_domains = np.array([[0, 2], [0, 2], [4, 6]])
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert not problem.is_inconsistent()

    def test_is_inconsistent_ok(self) -> None:
        shr_domains = np.array([[0, 2], [2, 0], [4, 6]])
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert problem.is_inconsistent()

    def test_is_not_solved_ok(self) -> None:
        shr_domains = np.array([[0, 2], [0, 2], [4, 6]])
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert problem.is_not_solved()

    def test_is_not_solved_ko(self) -> None:
        shr_domains = np.array([[0, 0], [2, 2], [6, 6]])
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets)
        assert not problem.is_not_solved()

    def test_filter(self) -> None:
        shr_domains = np.array([[0, 2], [0, 2], [4, 6]])
        dom_indices = [0, 1, 2]
        dom_offsets = [0, 0, 0]
        problem = Problem(shr_domains, dom_indices, dom_offsets, [Sum([2, 0, 1])])
        assert problem.filter()
        assert not problem.is_not_solved()
        assert np.all(problem.shr_domains == np.array([[2, 2], [2, 2], [4, 4]]))
