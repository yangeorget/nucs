import numpy as np

from ncs.propagators.affine_propagator import (
    OPERATOR_EQ,
    OPERATOR_GEQ,
    OPERATOR_LEQ,
    compute_domains,
)


class TestAffine:

    def test_compute_domains_EQ_(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([8, 1, 1, OPERATOR_EQ], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[1, 7], [1, 7]]))

    def test_compute_domains_EQ_2(self) -> None:
        domains = np.array([[5, 10], [5, 10], [5, 10]], dtype=np.int32)
        data = np.array([27, 1, 1, 1, OPERATOR_EQ], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[7, 10], [7, 10], [7, 10]]))

    def test_compute_domains_EQ_3(self) -> None:
        domains = np.array([[-2, -1], [2, 3]], dtype=np.int32)
        data = np.array([0, 1, 1, OPERATOR_EQ], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[-2, -2], [2, 2]]))

    def test_compute_domains_EQ_4(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([0, 1, -3, OPERATOR_EQ], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[3, 10], [1, 3]]))

    def test_compute_domains_EQ_5(self) -> None:
        domains = np.array([[-14, 11], [-4, 5]], dtype=np.int32)
        data = np.array([0, 1, 3, OPERATOR_EQ], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[-14, 11], [-3, 4]]))

    def test_compute_domains_LEQ(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([-1, 1, -1, OPERATOR_LEQ], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[1, 9], [2, 10]]))

    def test_compute_domains_GEQ(self) -> None:
        domains = np.array([[1, 10], [1, 10]], dtype=np.int32)
        data = np.array([1, 1, -1, OPERATOR_GEQ], dtype=np.int32)
        assert np.all(compute_domains(domains, data) == np.array([[2, 10], [1, 9]]))
