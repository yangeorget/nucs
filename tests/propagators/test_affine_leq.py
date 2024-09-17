import numpy as np

from nucs.constants import PROBLEM_INCONSISTENT, PROP_CONSISTENCY, PROP_ENTAILMENT
from nucs.numpy import new_data_by_values, new_shr_domains_by_values
from nucs.problems.problem import Problem
from nucs.propagators.affine_leq_propagator import compute_domains_affine_leq, get_triggers_affine_leq
from nucs.propagators.propagators import ALG_AFFINE_LEQ


class TestAffineLEQ:
    def test_get_triggers(self) -> None:
        data = new_data_by_values([1, -1, 8])
        assert np.all(get_triggers_affine_leq(2, data) == np.array([[True, False], [False, True]]))

    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(1, 10), (1, 10)])
        data = new_data_by_values([1, -1, -1])
        assert compute_domains_affine_leq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 9], [2, 10]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(1, 10), (1, 10)])
        data = new_data_by_values([1, 1, 8])
        assert compute_domains_affine_leq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 7], [1, 7]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(2, 3), (1, 2)])
        data = new_data_by_values([1, 1, 5])
        assert compute_domains_affine_leq(domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[2, 3], [1, 2]]))

    def test_filter(self) -> None:
        problem = Problem([(0, 2), (0, 2), (0, 2)])
        problem.add_propagators(
            [
                ([0, 1], ALG_AFFINE_LEQ, [-1, 1, -1]),
                ([1, 2], ALG_AFFINE_LEQ, [-1, 1, -1]),
                ([2, 0], ALG_AFFINE_LEQ, [-1, 1, -1]),
            ]
        )
        assert problem.filter(np.ones((3, 2), dtype=bool)) == PROBLEM_INCONSISTENT
