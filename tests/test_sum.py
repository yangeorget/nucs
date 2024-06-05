import numpy as np

from ncs.propagators.sum import Sum


class TestSum:
    def test_compute_domains_1(self) -> None:
        domains = np.array([[0, 1], [0, 1], [0, 1]])
        variables = [2, 0, 1]
        assert np.all(Sum(variables).compute_domains(domains) == np.array([[0, 2], [-1, 1], [-1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = np.array([[0, 2], [0, 2], [4, 6]])
        variables = [2, 0, 1]
        assert np.all(Sum(variables).compute_domains(domains) == np.array([[0, 4], [2, 6], [2, 6]]))
