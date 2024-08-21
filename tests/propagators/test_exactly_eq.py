import numpy as np

from ncs.memory import (
    PROP_CONSISTENCY,
    PROP_ENTAILMENT,
    PROP_INCONSISTENCY,
    new_data_by_values,
    new_domains_by_values,
)
from ncs.propagators.propagators import ALG_EXACTLY_EQ, compute_domains


class TestExactlyEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5])
        data = new_data_by_values([5, 1])
        assert compute_domains(ALG_EXACTLY_EQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 4], [3, 4], [3, 6], [6, 8], [3, 3], [5, 5]]))

    def test_compute_domains_2(self) -> None:
        domains = new_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5])
        data = new_data_by_values([5, 2])
        assert compute_domains(ALG_EXACTLY_EQ, domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 4], [3, 5], [3, 6], [6, 8], [3, 3], [5, 5]]))

    def test_compute_domains_3(self) -> None:
        domains = new_domains_by_values([(1, 4), (3, 5), (3, 6), (6, 8), 3, 5])
        data = new_data_by_values([5, 0])
        assert compute_domains(ALG_EXACTLY_EQ, domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_4(self) -> None:
        domains = new_domains_by_values([2, (0, 1), (3, 4), 2, 2])
        data = new_data_by_values([2, 3])
        assert compute_domains(ALG_EXACTLY_EQ, domains, data) == PROP_ENTAILMENT
        assert np.all(domains == np.array([[2, 2], [0, 1], [3, 4], [2, 2], [2, 2]]))
