import numpy as np

from ncs.propagators.propagators import ALG_AFFINE_LEQ, compute_domains


class TestAffineLEQ:
    def test_compute_domains_1(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([-1, 1, -1], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_LEQ, domains, data) == np.array([[1, 9], [2, 10]]))

    def test_compute_domains_2(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([8, 1, 1], dtype=np.int32)
        assert np.all(compute_domains(ALG_AFFINE_LEQ, domains, data) == np.array([[1, 7], [1, 7]]))
