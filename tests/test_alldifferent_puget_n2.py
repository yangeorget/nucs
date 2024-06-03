import numpy as np

from ncs.propagators.alldifferent_puget_n2 import AlldifferentPugetN2


class TestAlldiffentPugetN2:
    def test_compute_domains_1(self) -> None:
        domains = np.array([[3, 6], [3, 4], [2, 5], [2, 4], [3, 4], [1, 6]])
        variables = [0, 1, 2, 3, 4, 5]
        assert np.all(
            AlldifferentPugetN2(variables).compute_domains(domains)
            == np.array([[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]])
        )

    def test_compute_domains_2(self) -> None:
        domains = np.array([[0, 0], [2, 2], [1, 2]])
        variables = [0, 1, 2]
        assert np.all(AlldifferentPugetN2(variables).compute_domains(domains) == np.array([[0, 0], [2, 2], [1, 1]]))

    def test_compute_domains_3(self) -> None:
        domains = np.array([[0, 0], [0, 4], [0, 4], [0, 4], [0, 4]])
        variables = [0, 1, 2, 3, 4]
        assert np.all(
            AlldifferentPugetN2(variables).compute_domains(domains)
            == np.array([[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]])
        )
