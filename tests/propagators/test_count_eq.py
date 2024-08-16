import numpy as np

from ncs.memory import init_data_by_values, init_domains_by_values
from ncs.propagators.propagators import ALG_COUNT_EQ, compute_domains


class TestCountEQ:
    def test_compute_domains_1(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 1])
        data = init_data_by_values([5])
        assert compute_domains(ALG_COUNT_EQ, domains, data)
        assert np.all(domains == np.array([[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5], [1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 2])
        data = init_data_by_values([5])
        assert compute_domains(ALG_COUNT_EQ, domains, data)
        assert np.all(domains == np.array([[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]]))

    def test_compute_domains_3(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, 0])
        data = init_data_by_values([5])
        assert not compute_domains(ALG_COUNT_EQ, domains, data)

    def test_compute_domains_4(self) -> None:
        domains = init_domains_by_values([(1, 4), 5, (3, 6), (6, 8), 3, 5, (1, 2)])
        data = init_data_by_values([5])
        assert compute_domains(ALG_COUNT_EQ, domains, data)
        assert np.all(domains == np.array([[1, 4], [5, 5], [3, 6], [6, 8], [3, 3], [5, 5], [2, 2]]))

    def test_compute_domains_5(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5, (-1, 10)])
        data = init_data_by_values([5])
        assert compute_domains(ALG_COUNT_EQ, domains, data)
        assert np.all(domains == np.array([[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5], [1, 3]]))
