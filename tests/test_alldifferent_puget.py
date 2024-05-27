import numpy as np

from ncs.propagators.alldifferent_puget import AlldifferentPuget


class TestAlldiffentPuget:
    def test_compute_domains(self) -> None:
        domains = np.array([[3, 6], [3, 4], [2, 5], [2, 4], [3, 4], [1, 6]])
        variables = np.array([0, 1, 2, 3, 4, 5])
        assert np.all(
            AlldifferentPuget(variables).compute_domains(domains)
            == np.array([[3, 6], [3, 4], [2, 5], [2, 4], [3, 4], [1, 6]])
        )
