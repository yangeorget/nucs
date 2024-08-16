import numpy as np

from ncs.memory import init_data_by_values, init_domains_by_values
from ncs.propagators.propagators import ALG_COUNT_EQ, compute_domains, ALG_EXACTLY_EQ, ALG_MAX


class TestMax:
    def test_compute_domains_1(self) -> None:
        domains = init_domains_by_values([(1, 4), (2, 5), (0, 2)])
        data = init_data_by_values([])
        assert compute_domains(ALG_MAX, domains, data)
        assert np.all(domains == np.array([[1, 2], [2, 2], [2, 2]]))

    def test_compute_domains_2(self) -> None:
        domains = init_domains_by_values([(1, 4), (3, 5), (0, 2)])
        data = init_data_by_values([])
        assert not compute_domains(ALG_MAX, domains, data)

    def test_compute_domains_3(self) -> None:
        domains = init_domains_by_values([(2, 4), (2, 5), (0, 1)])
        data = init_data_by_values([])
        assert not compute_domains(ALG_MAX, domains, data)