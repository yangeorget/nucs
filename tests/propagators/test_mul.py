import numpy as np

from ncs.propagators.mul_propagator import compute_domains


class TestMul:

    def test_compute_domains_1(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array(0, dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[0, 0], [1, 10]]))

    def test_compute_domains_2(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array(3, dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[3, 10], [1, 3]]))

    def test_compute_domains_3(self) -> None:
        domains = np.array([[-14, 11], [-4, 5]], dtype=np.int32)
        data = np.array(-3, dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[-14, 11], [-3, 4]]))
