import numpy as np

from nucs.constants import PROP_CONSISTENCY, PROP_INCONSISTENCY
from nucs.numpy import new_data_by_values, new_shr_domains_by_values
from nucs.propagators.max_eq_propagator import compute_domains_max_eq


class TestMaxEQ:
    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(1, 4), (2, 5), (0, 2)])
        data = new_data_by_values([])
        assert compute_domains_max_eq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[1, 2], [2, 2], [2, 2]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(1, 4), (3, 5), (0, 2)])
        data = new_data_by_values([])
        assert compute_domains_max_eq(domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(2, 4), (2, 5), (0, 1)])
        data = new_data_by_values([])
        assert compute_domains_max_eq(domains, data) == PROP_INCONSISTENCY

    def test_compute_domains_4(self) -> None:
        domains = new_shr_domains_by_values([(0, 1), (0, 1), (0, 0)])
        data = new_data_by_values([])
        assert compute_domains_max_eq(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [0, 0], [0, 0]]))

    def test_compute_domains_5(self) -> None:
        domains = new_shr_domains_by_values([(0, 1), (0, 1), (2, 3)])
        data = new_data_by_values([])
        assert compute_domains_max_eq(domains, data) == PROP_INCONSISTENCY
