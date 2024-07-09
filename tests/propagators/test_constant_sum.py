import numpy as np

from ncs.propagators.constant_sum_propagator import compute_domains


class TestConstant_sum:

    def test_compute_domains_1(self) -> None:
        domains = np.array([[1, 10]], dtype=np.int32)
        data = np.array(8, dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[8, 8]]))

    def test_compute_domains_2(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array(8, dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[1, 7], [1, 7]]))
