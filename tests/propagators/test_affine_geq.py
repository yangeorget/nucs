import numpy as np

from ncs.memory import new_data_by_values, new_domains_by_values, PROP_CONSISTENCY, PROP_ENTAILMENT
from ncs.propagators.propagators import ALG_AFFINE_GEQ, compute_domains, get_triggers


class TestAffineGEQ:
    def test_get_triggers(self) -> None:
        data = new_data_by_values([1, -1, 8])
        assert np.all(get_triggers(ALG_AFFINE_GEQ, 2, data) == np.array([[False, True], [True, False]]))

    def test_compute_domains_1(self) -> None:
        domains = new_domains_by_values([(1, 10), (1, 10)])
        data = new_data_by_values([1, -1, 1])
        assert compute_domains(ALG_AFFINE_GEQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[2, 10], [1, 9]]))

    def test_compute_domains_2(self) -> None:
        domains = new_domains_by_values([(5, 10), (5, 10), (5, 10)])
        data = new_data_by_values([1, 1, 1, 27])
        assert compute_domains(ALG_AFFINE_GEQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[7, 10], [7, 10], [7, 10]]))

    def test_compute_domains_3(self) -> None:
        domains = new_domains_by_values([(5, 10), (1, 2)])
        data = new_data_by_values([1, 1, 6])
        assert compute_domains(ALG_AFFINE_GEQ, domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[5, 10], [1, 2]]))
