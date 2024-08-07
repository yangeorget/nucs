import numpy as np

from ncs.propagators.propagators import ALG_AFFINE_GEQ, compute_domains


class TestAffineGEQ:
    def test_compute_domains_1(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([1, 1, -1], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_GEQ, domains, data) == np.array([[2, 10], [1, 9]]))

    def test_compute_domains_2(self) -> None:
        domains = np.array([[5, 10], [5, 10], [5, 10]], dtype=np.int32)
        data = np.array([27, 1, 1, 1], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_GEQ, domains, data) == np.array([[7, 10], [7, 10], [7, 10]]))
