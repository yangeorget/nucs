import numpy as np

from nucs.memory import (
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    new_data_by_values,
    new_domains_by_values,
)
from nucs.propagators.propagators import ALG_MAX_LEQ, compute_domains


class TestMaxLEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_domains_by_values([(1, 4), (2, 5), (0, 2)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MAX_LEQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 2], [2, 2], [2, 2]]))

    def test_compute_domains_2(self) -> None:
        domains = new_domains_by_values([(1, 4), (3, 5), (0, 2)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MAX_LEQ, domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_3(self) -> None:
        domains = new_domains_by_values([(2, 4), (2, 5), (0, 1)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MAX_LEQ, domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_4(self) -> None:
        domains = new_domains_by_values([(0, 1), (0, 1), (2, 3)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MAX_LEQ, domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[0, 1], [0, 1], [2, 3]]))

    def test_compute_domains_5(self) -> None:
        domains = new_domains_by_values([(0, 1), (0, 1), (0, 0)])
        data = new_data_by_values([])
        assert compute_domains(ALG_MAX_LEQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [0, 0], [0, 0]]))