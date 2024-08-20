import numpy as np

from ncs.memory import new_data_by_values, new_domains_by_values
from ncs.propagators.propagators import ALG_AFFINE_EQ, compute_domains, get_triggers


class TestAffineEQ:
    def test_get_triggers(self) -> None:
        data = new_data_by_values([8, 1, -1])
        assert np.all(get_triggers(ALG_AFFINE_EQ, 2, data) == np.array([[True, True], [True, True]]))

    def test_compute_domains_1(self) -> None:
        domains = new_domains_by_values([(1, 10), (1, 10)])
        data = new_data_by_values([1, 1, 8])
        assert compute_domains(ALG_AFFINE_EQ, domains, data)
        assert np.all(domains == np.array([[1, 7], [1, 7]]))

    def test_compute_domains_2(self) -> None:
        domains = new_domains_by_values([(5, 10), (5, 10), (5, 10)])
        data = new_data_by_values([1, 1, 1, 27])
        assert compute_domains(ALG_AFFINE_EQ, domains, data)
        assert np.all(domains == np.array([[7, 10], [7, 10], [7, 10]]))

    def test_compute_domains_3(self) -> None:
        domains = new_domains_by_values([(-2, -1), (2, 3)])
        data = new_data_by_values([1, 1, 0])
        assert compute_domains(ALG_AFFINE_EQ, domains, data)
        assert np.all(domains == np.array([[-2, -2], [2, 2]]))

    def test_compute_domains_4(self) -> None:
        domains = new_domains_by_values([(1, 10), (1, 10)])
        data = new_data_by_values([1, -3, 0])
        assert compute_domains(ALG_AFFINE_EQ, domains, data)
        assert np.all(domains == np.array([[3, 10], [1, 3]]))

    def test_compute_domains_5(self) -> None:
        domains = new_domains_by_values([(-14, 11), (-4, 5)])
        data = new_data_by_values([1, 3, 0])
        assert compute_domains(ALG_AFFINE_EQ, domains, data)
        assert np.all(domains == np.array([[-14, 11], [-3, 4]]))
