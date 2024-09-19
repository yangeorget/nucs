import numpy as np

from nucs.constants import PROP_CONSISTENCY
from nucs.numpy import new_data_by_values, new_shr_domains_by_values
from nucs.propagators.alldifferent_propagator import compute_domains_alldifferent


class TestAlldifferent:

    def test_compute_domains_1(self) -> None:
        domains = new_shr_domains_by_values([(3, 6), (3, 4), (2, 5), (2, 4), (3, 4), (1, 6)])
        data = new_data_by_values([])
        assert compute_domains_alldifferent(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[6, 6], [3, 4], [5, 5], [2, 2], [3, 4], [1, 1]]))

    def test_compute_domains_2(self) -> None:
        domains = new_shr_domains_by_values([(0, 0), (2, 2), (1, 2)])
        data = new_data_by_values([])
        assert compute_domains_alldifferent(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [2, 2], [1, 1]]))

    def test_compute_domains_3(self) -> None:
        domains = new_shr_domains_by_values([(0, 0), (0, 4), (0, 4), (0, 4), (0, 4)])
        data = new_data_by_values([])
        assert compute_domains_alldifferent(domains, data) == PROP_CONSISTENCY
        assert np.all(domains == np.array([[0, 0], [1, 4], [1, 4], [1, 4], [1, 4]]))
