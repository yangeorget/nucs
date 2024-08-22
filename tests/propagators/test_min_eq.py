import numpy as np

from nucs.memory import (
    PROP_CONSISTENCY,
    PROP_INCONSISTENCY,
    new_data_by_values,
    new_domains_by_values,
)
from nucs.propagators.propagators import ALG_MIN_EQ, compute_domains


class TestMinEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_domains_by_values([(1, 4), (2, 5), (2, 6)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MIN_EQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[2, 4], [2, 5], [2, 4]]))

    def test_compute_domains_2(self) -> None:
        domains = new_domains_by_values([(1, 3), (3, 3), (4, 5)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MIN_EQ, domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_3(self) -> None:
        domains = new_domains_by_values([(2, 4), (2, 5), (6, 8)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MIN_EQ, domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_4(self) -> None:
        domains = new_domains_by_values([(0, 1), (0, 1), (1, 1)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MIN_EQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 1], [1, 1], [1, 1]]))

    def test_compute_domains_5(self) -> None:
        domains = new_domains_by_values([(2, 3), (2, 3), (0, 1)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MIN_EQ, domains, data) == PROP_INCONSISTENCY
