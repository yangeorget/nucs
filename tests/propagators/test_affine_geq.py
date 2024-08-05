import numpy as np

from ncs.propagators.affine_eq_propagator import compute_domains


class TestAffineGEQ:
    def test_compute_domains(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([1, 1, -1], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[2, 10], [1, 9]]))
