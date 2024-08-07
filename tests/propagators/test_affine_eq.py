import numpy as np

from ncs.propagators.propagators import ALG_AFFINE_EQ, compute_domains


class TestAffineEQ:

    def test_compute_domains_1(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([8, 1, 1], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_EQ, domains, data) == np.array([[1, 7], [1, 7]]))

    def test_compute_domains_2(self) -> None:
        domains = np.array([[5, 10], [5, 10], [5, 10]], dtype=np.int32)
        data = np.array([27, 1, 1, 1], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_EQ, domains, data) == np.array([[7, 10], [7, 10], [7, 10]]))

    def test_compute_domains_3(self) -> None:
        domains = np.array([[-2, -1], [2, 3]], dtype=np.int32)
        data = np.array([0, 1, 1], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_EQ, domains, data) == np.array([[-2, -2], [2, 2]]))

    def test_compute_domains_4(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([0, 1, -3], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_EQ, domains, data) == np.array([[3, 10], [1, 3]]))

    def test_compute_domains_5(self) -> None:
        domains = np.array([[-14, 11], [-4, 5]], dtype=np.int32)
        data = np.array([0, 1, 3], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_EQ, domains, data) == np.array([[-14, 11], [-3, 4]]))
