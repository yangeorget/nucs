import numpy as np

from ncs.memory import init_data_by_values, init_domains_by_values
from ncs.propagators.propagators import ALG_COUNT_EQ, compute_domains, ALG_EXACTLY_EQ


class TestExactlyEQ:
    def test_compute_domains_1(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5])
        data = init_data_by_values([5, 1])
        assert compute_domains(ALG_EXACTLY_EQ, domains, data)
        assert np.all(domains == np.array([[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5]]))

    def test_compute_domains_2(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5])
        data = init_data_by_values([5, 2])
        assert compute_domains(ALG_EXACTLY_EQ, domains, data)
        assert np.all(domains == np.array([[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5]]))

    def test_compute_domains_3(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5])
        data = init_data_by_values([5, 0])
        assert not compute_domains(ALG_EXACTLY_EQ, domains, data)
